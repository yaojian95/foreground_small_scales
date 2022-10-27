from crontab import CronTab

cron = CronTab(user=True)
job = cron.new(command='echo hello_world')
job.minute.every(1)
cron.write()



