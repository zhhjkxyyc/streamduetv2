from frontend.client import Client
from frontend.roi_client import RoIClient
class ClientFactory:
    @staticmethod
    def get_client(config, client_id, server=None, hname=None):
        if hname:
            # Initialize Client with hostname and client_id
            return Client(config, client_id, server_handle=hname)
        else:
            # Initialize Client with server and client_id
            return Client(config, client_id, server_handle=server)

    @staticmethod
    def get_roi_client(config, client_id, server=None, hname=None):
        if hname:
            # Initialize Client with hostname and client_id
            return RoIClient(config, client_id, server_handle=hname)
        else:
            # Initialize Client with server and client_id
            return RoIClient(config, client_id, server_handle=server)
