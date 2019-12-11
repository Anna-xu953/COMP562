
import json
import uuid
import requests
import datetime

def date_format(d):
    return d.strftime("%Y%m%d %H:%M:%S")

class LoggerException(Exception):
	def __str__(self):
		return "Logger exception occurred"



def default_logger_function(msg):
	print(msg)



## logging function implementations
class LogFunctor :
	params=None
	current_time=None
	logmsg=[]
	def __init__(self,params=None):
		self.params=params
		self.current_time=datetime.datetime
	def _timestamp(self):
		return date_format(self.current_time.now())
	def out(self,msg):
		print("%s : %s " % (self._timestamp(),msg))
		#print(msg)

class APILogFunctor (LogFunctor) :
	params=None
	logdocument={}
	credential={}
	userInfo={}
	baseurl=None
	token=None
	current_time=None
	header={}
	qparams={}
	def __init__(self,params):
		self.params=params
		self.credential=params["credential"]
		self.baseurl = params["baseurl"]
		self.current_time = datetime.datetime
		self.headers={'Content-Type' : 'application/json'}
		
		#print (self.credential)
		res=requests.post(self.baseurl + "/user_accounts/login", data= json.dumps(self.credential) ,headers=self.headers)
		if(res.status_code==200):			
			self.userInfo=res.json()
			self.token=self.userInfo["id"]
			self.qparams={'access_token' : self.token}
			self.logdocument={"id" : str(uuid.uuid1()), "name" : self.params["name"], "description" : self.params["description"] , "log" : [{"time" : self._timestamp(), "message" : "Logging Initialized" }] , "start_time" : self._timestamp(), "end_time" : None  }
			res=requests.post(self.baseurl+"/logs", data=json.dumps(self.logdocument), params=self.qparams , headers=self.headers)
			if(res.status_code==200):
				pass
			else:
				print("Log creation failed : " + `res.status_code`)
				raise LoggerException()

		else:
			raise LoggerException()

	def out(self,msg):
		self.logdocument["log"].append({"time": self._timestamp() , "message" : "%s" % str(msg)  })
		res=requests.put(self.baseurl+"/logs", data=json.dumps(self.logdocument), params=self.qparams, headers=self.headers)


### Logging function wrapper
class Logger :
	log_functors=[]
	params=None
	def __init__(self, lf=[LogFunctor()],params=None):
		self.log_functors=lf
		self.params=params
	def out(self,msg):
		for logelm in self.log_functors:
			logelm.out(msg)
