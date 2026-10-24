---
title: Celery Basics
tags: [python, backend]
date: 2025-03-05
---
# Celery Basics

Celery runs background tasks using a broker such as Redis or RabbitMQ.

## Celery beat

Celery beat is a scheduler that triggers periodic tasks, similar to cron.

## Retries

Use `autoretry_for` and exponential backoff to make tasks resilient.
