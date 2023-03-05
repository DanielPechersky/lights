scp lights.py ledpi:/home/pi/lights.py
# scp led-control.service ledpi:/etc/systemd/system/led-control.service
# ssh ledpi 'sudo systemctl daemon-reload'
ssh ledpi 'sudo systemctl restart led-control'
