

## смотреть логи
```bash
sudo journalctl -u stealth_celery.service -f
```

redis-cli -n 2 FLUSHDB && redis-cli -n 0 FLUSHDB && redis-cli -n 1 FLUSHDB && echo "Все базы Redis очищены"