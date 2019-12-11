import requests
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--host', dest='host', default='http://121.139.48.226:3100',
                         help='API Host', required=False)
    parser.add_argument('--id', dest='id', default=None,
                         help='Execution ID', required=True)
    parser.add_argument('--access_token', dest='access_token', default=None,
                         help='Access Token', required=True)
    args = parser.parse_args()
    params={"access_token" : args.access_token}
    uri = args.host + '/api/logs/' + args.id
    headers={'Content-Type' : 'application/json'}

    res=requests.get(uri, params=params, headers=headers)
    if res.status_code == 200 :
        logdoc=res.json()
        print("\n\n--------------------------------------------------------------------------------------------")
        print("Execution ID is         : %s" % logdoc["id"])
        print("--------------------------------------------------------------------------------------------")
        print("Application Name        : %s" % logdoc["name"])
        print("Application Description : %s" % logdoc["description"])
        print("--------------------------------------------------------------------------------------------\n\n")
        for elm in logdoc["log"]:
            print ("%s  :   %s" % (elm["time"],elm["message"]))

    else:
        print("Error with status code : %d " % res.status_code)

    print("\n\n Log dumped.")

if __name__ == '__main__' :
    main()
else:
    pass
