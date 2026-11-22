import time

from live_stt.supervisor import Child, Supervisor
from live_stt.settings import Settings


def test_crashing_child_restarts_with_backoff_and_healthy_child_stays():
    s = Settings()
    s.db.dsn = "postgresql://x:x@localhost:5499/none"
    sup = Supervisor(s, None, with_api=False)
    sup.children.clear()
    sup._add(Child("crasher", "selftest", "crash:0.2:7", 0))
    sup._add(Child("stable", "selftest", "sleep", 0))
    t_end = time.time() + 12
    while time.time() < t_end:
        sup.tick()
        time.sleep(0.1)
    c, st = sup.children["crasher"], sup.children["stable"]
    assert c.restarts >= 2 and c.last_exit == 7
    assert c.backoff >= 4  # exponential growth
    assert st.restarts == 0 and st.proc.is_alive()
    h = sup.health()
    assert h["children"]["stable"]["alive"] is True
    for ch in sup.children.values():
        sup._stop_child(ch)
    assert all(ch.proc is None for ch in sup.children.values())


def test_channel_sync_add_change_remove(monkeypatch):
    from live_stt.settings import ChannelConfig
    import live_stt.supervisor as supmod

    s = Settings()
    s.channels = [ChannelConfig(id="a", url="udp://1:1"), ChannelConfig(id="b", url="udp://1:2")]
    db = [ChannelConfig(id="b", url="udp://1:9"), ChannelConfig(id="c", url="udp://1:3"),
          ChannelConfig(id="d", url="udp://1:4", enabled=False)]
    monkeypatch.setattr(supmod, "fetch_db_channels", lambda dsn: db)
    sup = Supervisor(s, None, with_api=False)
    sup.sync_channels()
    workers = {n: c for n, c in sup.children.items() if c.kind == "worker"}
    assert set(workers) == {"worker:a", "worker:b", "worker:c"}  # DB overrides b, d disabled
    assert '"udp://1:9"' in workers["worker:b"].arg
    ports = {c.port for c in workers.values()}
    assert len(ports) == 3
    db[:] = [ChannelConfig(id="b", url="udp://1:10")]
    s.channels = []
    sup.sync_channels()
    workers = {n: c for n, c in sup.children.items() if c.kind == "worker"}
    assert set(workers) == {"worker:b"} and '"udp://1:10"' in workers["worker:b"].arg
