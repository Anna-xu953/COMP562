
import json

def load_config_file(filename):
    with open(filename) as cfg:
        params=json.load(cfg)
        assert(params["config_type"]=="app.configuration" ) , "Config file is not correct app.configuration type"
        for key in params :
        	for elmkey in params[key]:
        		if key=="directories"  or key=="etc" or key=="files":
        			#print(key,elmkey)
        			#print(params[key][elmkey])
        			if params[key][elmkey]["type"]=="string":
        				params[key][elmkey]["value"]=str(params[key][elmkey]["value"])
        return params