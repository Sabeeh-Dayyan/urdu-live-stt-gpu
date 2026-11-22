"""Kafka / Redpanda bus adapter — STUB (interface + mapping, not implemented in v1).

Mapping to the Bus contract when implemented (e.g. with confluent-kafka):
* topic per stream name; partition key = ``channel_id`` (keeps per-channel ordering and gives the
  ASR consumer channel affinity needed by the local_agreement streaming sessions);
* ``consume(group, consumer)`` → consumer group subscription; ``ack`` → commit offsets after the
  event is published (at-least-once; consumers are idempotent via segment_id upserts);
* audio payloads in the message value as int16 PCM with JSON metadata in headers;
* ``broadcast`` → a compacted/short-retention ``transcripts.live`` topic or keep Redis pub/sub
  for WebSocket fan-out;
* ``set_value``/``get_value`` (health, control) → keep Redis, or a compacted topic.

Selected by ``bus.backend: kafka`` once implemented; until then constructing it raises.
"""

from __future__ import annotations

from live_stt.bus.base import Bus, Message


class KafkaBus(Bus):
    def __init__(self, bootstrap_servers: str, **kwargs) -> None:
        self.bootstrap_servers = bootstrap_servers
        raise NotImplementedError(
            "KafkaBus is a v1 stub; use RedisStreamsBus (see module docstring for the mapping)")

    def publish(self, stream: str, msg: Message) -> str:  # pragma: no cover - stub
        raise NotImplementedError

    def consume(self, stream, group, consumer, count=10, block_ms=0):  # pragma: no cover
        raise NotImplementedError

    def ack(self, stream, group, ids):  # pragma: no cover
        raise NotImplementedError

    def broadcast(self, channel, msg):  # pragma: no cover
        raise NotImplementedError

    def depth(self, stream, group):  # pragma: no cover
        raise NotImplementedError
