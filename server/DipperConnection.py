import websocket
import urllib.request
import urllib.parse
import json
import http.client
import http.cookiejar


class DipperConnection:
    def __init__(self, host, username, password):
        self.host_ = host
        self.base_url = 'http://'+host
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cj))
        url = self.base_url+'/common/login?' + \
            urllib.parse.urlencode(
                {'username': username, 'password': password, 'product':'dipper.training'})
        # print(url)
        res = self.opener.open(
            urllib.request.Request(url, method='GET')).read()
        data=json.loads(res.decode())
        # print(self.cj.__dict__['_cookies'])
        #self.session_id_ = self.cj.__dict__[
            #'_cookies'][host]['/']['session_id'].value
        self.session_id_ = data.get('session_id','')
        print("Loggin success: "+self.session_id_)

    #pip install websocket-client==0.48.0
    def send_ws(self, url, socket_string):
        ws_url = 'ws://'+self.host_+":80"+url
        self.ws = websocket.WebSocket(
            fire_cont_frame=True, skip_utf8_validation=True, enable_multithread=True)
        # self.ws=websocket.WebSocket(skip_utf8_validation=True,enable_multithread=True)
        self.ws.connect(ws_url, cookie="session_id=" + self.session_id_)

        frame_size = 1024 * 1024
        count = socket_string.__len__()
        index = 0 | 0
        if count > frame_size:
            frame_data = socket_string[index:index + frame_size]
            frame = websocket.ABNF(
                data=frame_data, fin=0, opcode=websocket.ABNF.OPCODE_BINARY)
            self.ws.send_frame(frame)
            index += frame_size
            while count > index + frame_size:
                frame_data = socket_string[index:index + frame_size]
                frame = websocket.ABNF(
                    data=frame_data, fin=0, opcode=websocket.ABNF.OPCODE_CONT)
                self.ws.send_frame(frame)
                index += frame_size
            frame_data = socket_string[index:]
            frame = websocket.ABNF(
                data=frame_data, fin=1, opcode=websocket.ABNF.OPCODE_CONT)
            self.ws.send_frame(frame)

        else:
            frame_data = socket_string[index:index + frame_size]
            frame = websocket.ABNF(
                data=frame_data, fin=1, opcode=websocket.ABNF.OPCODE_BINARY)
            self.ws.send_frame(frame)
        fin = 0
        data = b''
        while fin == 0:
            result = self.ws.recv_data_frame(True)
            data += result[1].data
            fin = result[1].fin
        return data

    def send_post(self, method, data):
        try:
            url = self.base_url+method
            print(url)
            params = bytes(json.dumps(data), 'utf-8')
            req = urllib.request.Request(url, params, method='POST')
            res = self.opener.open(req).read()
            return res
        except BaseException as exp:
            print('exception occurred:'+exp)
            raise exp

    def send_get(self, method, data):
        try:
            url = self.base_url+method
            if data is not None:
                url += '?'
                for k,v in data.items():
                    url += "{0}={1}&".format(k, v)
            if url.endswith('&'):
                url = url[:-1]
            print(url)
            req = urllib.request.Request(url, method='GET')
            res = self.opener.open(req).read()
            return res
        except BaseException as exp:
            print('exception occurred:'+exp)
            raise exp
