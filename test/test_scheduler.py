from DeSSL import SCHEDULER_REGISTRY
from pytest import raises


def test_linear_scheduler():
    scheduler = SCHEDULER_REGISTRY('linear')(5, 10)
    assert scheduler() == 5
    scheduler.step()
    assert scheduler() == 15


def test_identity_scheduler():
    scheduler = SCHEDULER_REGISTRY('linear')(5)
    assert scheduler() == 5
    scheduler.step()
    assert scheduler() == 5


def test_lambda_scheduler():
    scheduler = SCHEDULER_REGISTRY('lambda')(lambda x: x + 1)
    assert scheduler() == 1
    scheduler.step()
    assert scheduler() == 2


def test_import_scheduler():
    from DeSSL.scheduler import Lambda
    scheduler = Lambda(lambda x: x + 1, 100)
    scheduler.step()
    assert scheduler() == 102
    with raises(ImportError):
        from DeSSL import Lambda
