
import caduceus.core.logger as logger

class Learnable :
    _params={}
    _app_name=""
    _loggers=logger.Logger()
    def __init__(self,params):
        self._params=params       
    def _initialize(self):
        if('logger' in self._params):
            self._loggers=self._params["logger"]
        else:
            self._loggers=logger.Logger()
    def log(self,msg):
        self._loggers.out(msg)
    def get_params(self):
        return self._params
    def get_app_name(self):
        return self._app_name
    def train(self,train_params=_params):
        pass
    def test(self, test_params=_params):
        pass




