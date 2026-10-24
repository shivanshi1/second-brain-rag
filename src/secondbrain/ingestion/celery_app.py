from celery import Celery

from secondbrain.config import settings

celery_app = Celery(
    "secondbrain",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["secondbrain.ingestion.tasks"],
)
celery_app.conf.update(task_track_started=True, worker_prefetch_multiplier=1)
