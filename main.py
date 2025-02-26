import json
import os

import pm4py
from flask import Flask
from kafka import KafkaConsumer
from prometheus_client import make_wsgi_app, Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import time
from threading import Thread

from process_mining_core.datastructure.core.event import Event
from process_mining_core.datastructure.core.event_log import SerializableEventLog
from process_mining_core.datastructure.core.model.petri_net import SerializablePetriNet
from process_monitor.petri_net_serdes import PetriNetSerDes

#from pm4py.visualization.petri_net import visualizer

app = Flask(__name__)

def env_or_default(env_key, default_value):
    if env_key in os.environ:
        return os.environ[env_key]
    else:
        print("Warning! Env not set using default value")
        return default_value

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

log_topic = env_or_default("LOG_TOPIC", "input")
model_topic = env_or_default("MODEL_TOPIC", "model")
bootstrap_server = env_or_default("BOOTSTRAP_SERVER", "minikube:32207")

log_consumer = KafkaConsumer(
        log_topic,
        bootstrap_servers=bootstrap_server,
        group_id="process-monitor")

model_consumer = KafkaConsumer(
        model_topic,
        bootstrap_servers=bootstrap_server,
        group_id="process-monitor2")

PRECISION_TBR = Gauge(
'precision_tbr',
    'Precision using token based replay',
)

REPLAY_FITNESS = Gauge(
'fitness_tbr',
    'Fitness using token based replay',
)

F1_SCORE = Gauge(
'f1_tbr',
    'F1-Score',
)

window_size = 50
event_log_size = 100
evaluation_interval = 5

def _transform_record_to_event(record):
    record_dict = json.loads(record.value.decode())
    return Event(
        timestamp=record_dict["timestamp"],
        activity=record_dict["activity"],
        node=record_dict["node"],
        case_id=record_dict["caseid"],
        group_id=record_dict["group"]
    )

def get_event_log():
    event_log_result = log_consumer.poll()
    log = []
    for partition in event_log_result:
        consumed_records = event_log_result[partition]
        log.extend([_transform_record_to_event(record) for record in consumed_records])
    event_log = SerializableEventLog(log)
    print(len(log))
    return event_log.to_pm4py_event_log()


def get_model():
    model_result = model_consumer.poll()
    petri_net_des: SerializablePetriNet | None = None
    for partition in model_result:
        consumed_records = model_result[partition]
        petri_net_des = PetriNetSerDes().deserialize(consumed_records[-1:][0].value.decode())
    if petri_net_des:
        return petri_net_des.to_pm4py_petri_net()
    else:
        return None, None, None

def _window_results(precision_results):
    if len(precision_results) > window_size:
        windowed_results = precision_results[-window_size:]
    else:
        windowed_results = precision_results

    if len(windowed_results):
        average_precision = sum(windowed_results) / len(windowed_results)
    else:
        average_precision = 0
    return average_precision

def update_precision(precision_results, event_log, petri_net, initial_marking, final_marking):
    precision = pm4py.precision_token_based_replay(event_log, petri_net, initial_marking, final_marking)
    precision_results.append(precision)
    windowed_precision = _window_results(precision_results)
    PRECISION_TBR.set(windowed_precision)
    print(windowed_precision)

def update_fitness(fitness_results, event_log, petri_net, initial_marking, final_marking):
    fitness = pm4py.fitness_token_based_replay(event_log, petri_net, initial_marking, final_marking)
    fitness_results.append(fitness['average_trace_fitness'])
    windowed_fitness = _window_results(fitness_results)
    REPLAY_FITNESS.set(windowed_fitness)
    print(windowed_fitness)

def get_accuracy():
    precision_results = []
    fitness_results = []
    while True:
        event_log = get_event_log()
        petri_net, initial_marking, final_marking = get_model()

        if not event_log.empty and petri_net:
            #gviz = visualizer.apply(petri_net, initial_marking, final_marking)
            #visualizer.view(gviz)
            update_precision(precision_results, event_log, petri_net, initial_marking, final_marking)
            update_fitness(fitness_results, event_log, petri_net, initial_marking, final_marking)
        time.sleep(evaluation_interval)



@app.route('/')
def index():
    return "Accuracy Monitor"

if __name__ == '__main__':
    thread = Thread(target=get_accuracy)
    thread.start()
    app.run(host='0.0.0.0', port=5000)