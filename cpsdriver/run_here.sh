TOKEN="9e8864c9-ece9-4d6b-9aad-eea944ce371e"
DB_ADDRESS="mongodb+srv://cpsweek:Rn6ubiLFhiIIriTq@team5-aifi-comp-hxuhd.mongodb.net"
API_ADDRESS="http://aifi.io/cpsweek/api/v1"
COMMAND="BASELINE-1"
SAMPLE="nodepth"
AIFI_CPSWEEK_COMP__TOKEN=$TOKEN
AIFI_CPSWEEK_COMP__DB_ADDRESS=$DB_ADDRESS
AIFI_CPSWEEK_COMP__API_ADDRESS=$API_ADDRESS
AIFI_CPSWEEK_COMP__COMMAND=$COMMAND
AIFI_CPSWEEK_COMP__SAMPLE=$SAMPLE
#python main.py
python3 old_main.py --command $COMMAND --sample $SAMPLE --db-address $DB_ADDRESS --api-address $API_ADDRESS --token $TOKEN
#python3 main_thread.py
