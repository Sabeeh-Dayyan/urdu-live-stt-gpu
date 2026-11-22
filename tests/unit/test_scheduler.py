import random
import threading

from live_stt.asr.scheduler import FAST, REFINE, Job, Scheduler


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def job(i, p=FAST, dur=5.0, t=1000.0, key="", partial=False, recv=0.0):
    return Job(str(i), p, dur, t, key, partial, payload={"i": i}, t_received=recv)


def test_fast_before_refine_and_refine_only_when_fast_empty():
    s = Scheduler(wall=Clock())
    s.push(job(1, REFINE))
    s.push(job(2, FAST))
    s.push(job(3, FAST))
    order = []
    for _ in range(3):
        d = s.next()
        order.append((d.pass_, d.jobs[0].msg_id))
    assert order == [(FAST, "2"), (FAST, "3"), (REFINE, "1")]
    assert s.next().jobs == []


def test_refine_lag_skips_with_reason():
    w = Clock(2000.0)
    s = Scheduler(refine_max_lag_s=300, wall=w)
    s.push(job(1, REFINE, t=1000.0))  # 1000 s old
    s.push(job(2, REFINE, t=1990.0))
    d = s.next()
    assert [(j.msg_id, r) for j, r in d.skipped] == [("1", "refine_lag")]
    assert d.pass_ == REFINE and d.jobs[0].msg_id == "2"


def test_refine_queue_full_drops_oldest():
    s = Scheduler(refine_max=2, wall=Clock())
    for i in range(3):
        s.push(job(i, REFINE))
    d = s.next()
    assert [(j.msg_id, r) for j, r in d.skipped] == [("0", "refine_queue_full")]
    assert s.space(REFINE) == 1


def test_fast_never_dropped_even_over_capacity():
    s = Scheduler(fast_max=2, wall=Clock())
    for i in range(10):
        s.push(job(i))
    assert s.depth(FAST) == 10 and s.space(FAST) == 0
    got = [s.next().jobs[0].msg_id for _ in range(10)]
    assert got == [str(i) for i in range(10)]


def test_partial_superseded_by_newer_partial_and_final():
    s = Scheduler(wall=Clock())
    s.push(job(1, key="k", partial=True))
    s.push(job(2, key="k", partial=True))
    s.push(job(3, key="other", partial=True))
    s.push(job(4, key="k", partial=False))  # final
    d = s.next()
    assert sorted((j.msg_id, r) for j, r in d.skipped) == [("1", "superseded"), ("2", "superseded")]
    ids = [d.jobs[0].msg_id] + [s.next().jobs[0].msg_id for _ in range(1)]
    assert ids == ["3", "4"]


def test_batching_waits_and_groups_by_length():
    mono = Clock(0.0)
    s = Scheduler(max_batch=3, max_wait_s=0.05, batched=True, wall=Clock(), mono=mono)
    s.push(job(1, dur=5.0, recv=0.0))
    assert s.next().jobs == []  # waits for more
    mono.t = 0.06
    d = s.next()
    assert [j.msg_id for j in d.jobs] == ["1"]
    for i, dur in enumerate([5.0, 1.0, 5.2, 9.0, 4.9], start=10):
        s.push(job(i, dur=dur, recv=mono.t))
    d = s.next()  # 5 queued ≥ max_batch → immediate
    assert [j.msg_id for j in d.jobs] == ["10", "14", "12"]  # 5.0 then closest 4.9, 5.2
    assert s.depth(FAST) == 2


def test_concurrent_push_and_consume_accounts_for_everything():
    s = Scheduler(refine_max=50, refine_max_lag_s=1e9)
    N = 2000
    seen, skipped = [], []

    def producer(base):
        for i in range(N // 4):
            p = FAST if random.random() < 0.6 else REFINE
            partial = p == FAST and random.random() < 0.3
            s.push(job(f"{base}-{i}", p, key=f"k{random.randint(0, 20)}" if partial else "",
                       partial=partial, t=__import__("time").time()))

    threads = [threading.Thread(target=producer, args=(b,)) for b in range(4)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads) or s.depth(FAST) or s.depth(REFINE):
        d = s.next()
        seen += [j.msg_id for j in d.jobs]
        skipped += [j.msg_id for j, _ in d.skipped]
    d = s.next()
    seen += [j.msg_id for j in d.jobs]
    skipped += [j.msg_id for j, _ in d.skipped]
    assert len(seen) + len(skipped) == N
    assert len(set(seen) | set(skipped)) == N


def test_packing_fifo_up_to_budget_and_lone_job_not_delayed():
    s = Scheduler(pack_s=28, wall=Clock())
    s.push(job(1, dur=8))
    d = s.next()
    assert [j.msg_id for j in d.jobs] == ["1"]  # alone, immediately
    for i, dur in enumerate([8, 8, 8, 8, 3], start=10):
        s.push(job(i, dur=dur))
    d = s.next()
    assert [j.msg_id for j in d.jobs] == ["10", "11", "12"]  # 8+.3+8+.3+8 = 24.6 ≤ 28 < 32.9
    d = s.next()
    assert [j.msg_id for j in d.jobs] == ["13", "14"]


def test_packing_never_includes_partials_or_stream_jobs():
    s = Scheduler(pack_s=28, wall=Clock())
    s.push(job(1, dur=3))
    s.push(job(2, dur=3, key="k", partial=True))
    s.push(job(3, dur=3))
    assert [j.msg_id for j in s.next().jobs] == ["1"]
    assert [j.msg_id for j in s.next().jobs] == ["2"]
    j = job(4, dur=1)
    j.payload = {"kind": "stream"}
    s.push(j)
    s.push(job(5, dur=1))
    assert [x.msg_id for x in s.next().jobs] == ["3"]
    assert [x.msg_id for x in s.next().jobs] == ["4"]
