import imp
import tensorflow as tf
import json
import requests
import uuid 
import datetime 
import argparse
import caduceus.core.io as cio 
import caduceus.core.logger as logger

def main(_) :

    #parsing command line
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config_filename', dest='config_filename', default=None,
                         help='application configuration file', required=True)
    args = parser.parse_args()
    params=cio.load_config_file(args.config_filename)
    #print(params)

    # Initialize logging
    # apiexplorerurl="http://121.139.48.226:3100/explorer"
    # apiurl="http://121.139.48.226:3100/api/"
    # credential = {'username':'admin', 'password':'caduceus123!', 'email' : 'admin@caduceus.co.kr'}
    # headers={'Content-Type' : 'application/json'}
    # logparams={"credential" : credential, "baseurl" : apiurl , "headers" : headers,  "name" : params["application"]["application_name"] , "description" : params["application"]["application_description"]}
    lf1=logger.LogFunctor()
    #lf2=logger.APILogFunctor(logparams)
    #lg=logger.Logger([lf1,lf2])
    lg=logger.Logger([lf1])
    
    print("\n\n--------------------------------------------------------------------------------------------")
    # print("Execution ID is         : %s " % lf2.logdocument["id"])
    # print("--------------------------------------------------------------------------------------------")
    # print("Log server explorer  is : %s" % apiexplorerurl)
    # print("Access Token is         : %s " % lf2.token)
    # print("--------------------------------------------------------------------------------------------")
    print("Application Name        : %s " % params["application"]["application_name"])
    print("Application Description : %s" % params["application"]["application_description"])
    print("Application module_file : %s" % params["application"]["module_file"])
    print("--------------------------------------------------------------------------------------------\n\n")

    app_module = imp.load_source(params["application"]["module_name"],
                                params["application"]["module_file"])

    params["logger"] = lg
    app=app_module._LearnerImpl(params)
    if (params["training_parameters"]["testing"]["value"] == False):
        app.train()
    else:
        app.test()

if __name__ == '__main__' :

    tf.app.run()


else :
    pass
