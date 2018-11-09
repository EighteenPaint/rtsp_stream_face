from kafka import KafkaProducer
import json
producer = KafkaProducer(bootstrap_servers=['master:9092', 'node1:9092', 'node2:9092'])
def sendMessage(topic, message):
    producer.send(topic, json.dumps(message, ensure_ascii=False))
