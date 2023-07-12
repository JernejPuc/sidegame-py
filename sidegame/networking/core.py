"""Supporting elements for the predictive client-authoritative server model"""

import struct
import socket
from select import select
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Tuple

from sidegame.utils import MovingAverageTracker


class Entry:
    """
    An atomic descriptor that the client uses to partially infer the state
    of a larger record of organised information on the server,
    i.e. details of initialisation, log history, or game state
    at a specific point in time.
    """

    TYPE_INIT = 0
    TYPE_LOG = 1
    TYPE_STATE = 2

    def __init__(self, id_: int, type_: int, counter: int, timestamp: float, data: Any):
        self.id = id_
        self.type = type_
        self.counter = counter
        self.timestamp = timestamp
        self.data = data


class Action:
    """
    A singular action of a client at a specific point in time,
    corresponding to an initialisation request, a command that interacts with
    or controls the simulation or the server process itself,
    or an IO (mouse/keyboard/view) state report.

    Includes some attributes that help with client-side prediction.
    """

    TYPE_INIT = 0
    TYPE_LOG = 1
    TYPE_STATE = 2

    def __init__(self, type_: int, counter: int, timestamp: float, data: Any, dt: float = float('nan')):
        self.type = type_
        self.counter = counter
        self.timestamp = timestamp
        self.data = data
        self.dt = dt
        self.processed = False


class EventBase:
    """An extendable container for in-simulation event types and structures."""

    EMPTY_EVENT_LIST: Iterable['EventBase'] = []

    def __init__(self, type_: int, data: Any):
        self.type = type_
        self.data = data


class Entity:
    """
    A representation of a client in the simulation.
    Includes attributes for synchronisation of client-server representations.
    """

    def __init__(self, id_: int):
        self.id = id_
        self.local_log_counter = -1
        self.global_log_counter = -1
        self.action_counter = -1
        self.confirmed_action_counter = -1
        self.states: Deque[Entry] = deque()
        self.actions: Deque[Action] = deque()


class Node:
    """
    A representation of a client in the network.
    Allows tracking of latency and setting the interpolation window
    to facilitate lag compensation methods.
    """

    DISCONNECTION_TIMEOUT = 4.

    def __init__(self, id_: int = None, address: Tuple[str, int] = None, tracker: MovingAverageTracker = None):
        self.id = id_
        self.address = address
        self.tracker = tracker
        self.interp_window = 0.

        self.incoming_buffer: Deque[bytes] = deque()
        self.outgoing_buffer: Deque[bytes] = deque()

        self.time_of_last_contact: float = None

    def update_connection_status(self, timestamp: float):
        """Update time of last contact with the network."""

        self.time_of_last_contact = timestamp

    def verify_connection(self, timestamp: float):
        """
        Raise a disconnection message after exchange with the node-in-contact
        times out.
        """

        if timestamp - self.time_of_last_contact >= self.DISCONNECTION_TIMEOUT:
            raise ConnectionError('Connection timed out.')


class Recorder:
    """
    For accumulating, reading, and storing network data (in byte strings)
    as binary records.
    """

    _MIN_CHUNK_SIZE = 200000
    _MIN_SQUEEZE_LEN = 500
    _ENTRY_SEPARATOR: str = b'\x01\x1e\x1e\x1e\x1e\x17'

    SOURCE_UNSORTED = 0
    SOURCE_SERVER = 1
    SOURCE_CLIENT = 2

    def __init__(self, file_path: str = None):
        self.counter = 0
        self.timestamp = 0.
        self.chunks: Deque[bytes] = deque()
        self.buffer: Deque[bytes] = deque()
        self.record: bool = file_path is not None
        self.file_path = file_path

    def update_meta(self, timestamp: float):
        """Update counter and timestamp that will contextualise the following data."""

        self.counter += 1
        self.timestamp = timestamp

    def add_meta(self, data: bytes, source: int) -> bytes:
        """
        Prepend timestamp, counter, and source to a single packet of data
        to allow it to be categorised upon review.
        """

        return struct.pack('>fLB', self.timestamp, self.counter, source) + data

    @staticmethod
    def split_meta(data: bytes) -> Tuple[Tuple[float, int, int], bytes]:
        """Split off timestamp, counter, and source from a single packet of data."""

        return struct.unpack('>fLB', data[:9]), data[9:]

    def append(self, data: bytes, source: int = SOURCE_UNSORTED):
        """Conditionally add a single packet of data to the recording buffer."""

        if self.record:
            self.buffer.append(self.add_meta(data, source))

    def extend(self, data: Iterable[bytes], source: int = SOURCE_UNSORTED):
        """Conditionally add multiple packets of data to the recording buffer."""

        if self.record:
            self.buffer.extend(self.add_meta(datum, source) for datum in data)

    def squeeze(self, all_: bool = False):
        """Merge currently buffered data into a single larger chunk."""

        if all_ or len(self.buffer) >= self._MIN_SQUEEZE_LEN:
            squeezed_entries = self._ENTRY_SEPARATOR.join(self.buffer)
            self.buffer.clear()
            self.buffer.append(squeezed_entries)

    def cache_chunks(self):
        """
        Move a portion of the recording buffer to a buffer of larger chunks,
        which are prevented from being repeatedly squeezed.
        """

        while self.buffer and len(self.buffer[0]) >= self._MIN_CHUNK_SIZE:
            self.chunks.appendleft(self.buffer.popleft())

    def restore_chunks(self):
        """Move cached chunks back to the main recording buffer."""

        self.buffer.extendleft(self.chunks)
        self.chunks.clear()

    def read(self):
        """Extend the recording buffer by reading a recording file."""

        if self.file_path is None:
            raise ValueError('Path to recording is not set.')

        self.buffer.extend(self.read_into_buffer(self.file_path))

    def save(self):
        """Dump the current recording buffer into a recording file."""

        if self.file_path is None:
            raise ValueError('Path to recording is not set.')

        self.save_from_buffer(self.file_path, self.buffer)

    @staticmethod
    def read_into_buffer(file_path: str) -> List[bytes]:
        """Read a record file and return its separated entries."""

        with open(file_path, 'rb') as recording_file:
            record = recording_file.read().split(Recorder._ENTRY_SEPARATOR)

        return record

    @staticmethod
    def save_from_buffer(filename: str, record: Iterable[bytes]):
        """Dump a sequence of separated entries into a record file."""

        with open(filename, 'wb') as record_file:
            record_file.write(Recorder._ENTRY_SEPARATOR.join(record))


class ClientSocket:
    """
    An extension of the UDP socket to enable polling of readable state,
    handle some network and client characteristics, and exchange data
    with its server counterpart in a predictable way.

    NOTE: UDP is preferred for fast-paced real-time games due to lower latency.
    Packet loss and ordering can be handled by a custom abstraction.
    """

    _MAX_CLI_MSG_CHUNK_SIZE = 512
    _MAX_SRV_MSG_CHUNK_SIZE = 1024

    def __init__(self, server_address: Tuple[str, int], client_message_size: int, server_message_size: int):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.connect(server_address)
        self._socket.settimeout(0.)

        self._server_message_size = server_message_size
        self._recv_chunk_size = (self._MAX_SRV_MSG_CHUNK_SIZE // server_message_size) * server_message_size
        self._send_chunk_size = (self._MAX_CLI_MSG_CHUNK_SIZE // client_message_size) * client_message_size

        self._readable_sockets: List[socket.socket] = [self._socket]
        self._placeholder_sockets: List[socket.socket] = []

        self.node = Node()
        self._incoming_buffer: Deque[bytes] = self.node.incoming_buffer
        self._outgoing_buffer: Deque[bytes] = self.node.outgoing_buffer

    def close(self):
        """Shutdown and close the underlying UDP socket."""

        self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()

    def redirect(self, new_server_addres: Tuple[str, int]):
        """
        Close the socket linked to the current server address
        and reinitialise it, linking it to the new one.
        """

        self.close()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.connect(new_server_addres)
        self._socket.settimeout(0.)
        self._readable_sockets[0] = self._socket

    def sendall(self, data: bytes):
        """
        Ensure that the entirety of the data is sent
        (in chunks of limited size).

        NOTE: Joint packets could otherwise result in size that is
        unpredictably larger than the receiving buffer, but the latter
        must be large enough to be able to read full datagrams.
        """

        if len(data) > self._send_chunk_size:
            for i in range(0, len(data), self._send_chunk_size):
                self._socket.sendall(data[i:i+self._send_chunk_size])

        else:
            self._socket.sendall(data)

    def send(self, timestamp: float = 0.):
        """Send and clear buffered data after confirming connection to the server."""

        self.node.verify_connection(timestamp)

        self.sendall(b''.join(self._outgoing_buffer))

        self._outgoing_buffer.clear()

    def recv(self, timestamp: float = 0.):
        """
        Check if data is available for reading and receive it into a buffer.
        A successful receive also updates the connection status.
        """

        while True:
            readable, _, _ = select(self._readable_sockets, self._placeholder_sockets, self._placeholder_sockets, 0.)

            if readable:
                data = self._socket.recv(self._recv_chunk_size)

                # Drop incomplete data packets
                if len(data) % self._server_message_size:
                    if len(data) < self._server_message_size:
                        continue

                    data = data[:len(data) - len(data) % self._server_message_size]

                # Separate individual packets
                if len(data) > self._server_message_size:
                    self._incoming_buffer.extend(
                        data[i:i+self._server_message_size] for i in range(0, len(data), self._server_message_size))
                else:
                    self._incoming_buffer.append(data)

                self.node.update_connection_status(timestamp)

            else:
                break

    def exchange(self, request: bytes, timestamp: float = 0.) -> bytes:
        """Exchange a single packet of data with the server."""

        # Temporarily set blocking
        self._socket.settimeout(self.node.DISCONNECTION_TIMEOUT)

        self.sendall(request)

        reply = self._socket.recv(self._server_message_size)

        assert len(reply) == self._server_message_size, \
            f'Expected a single packet of size {self._server_message_size}, got {len(reply)}.'

        self.node.update_connection_status(timestamp)

        # Unblock
        self._socket.settimeout(0.)

        return reply


class ServerSocket:
    """
    An extension of the UDP socket to enable polling of readable state,
    handle some network and server characteristics, and exchange data
    with its client counterparts in a predictable way.

    Allows tracking of latency per client node to facilitate
    lag compensation methods.
    """

    _MAX_CLI_MSG_CHUNK_SIZE = 512
    _MAX_SRV_MSG_CHUNK_SIZE = 1024
    _MAX_NUM_CONCURRENT_CONNECTIONS = 25

    def __init__(
        self,
        binding_address: Tuple[str, int],
        client_message_size: int,
        server_message_size: int,
        latency_window_length: int
    ):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(binding_address)
        self._socket.settimeout(0.)

        self._client_message_size = client_message_size
        self._recv_chunk_size = (self._MAX_CLI_MSG_CHUNK_SIZE // client_message_size) * client_message_size
        self._send_chunk_size = (self._MAX_SRV_MSG_CHUNK_SIZE // server_message_size) * server_message_size
        self._latency_window_length = latency_window_length

        self._readable_sockets: List[socket.socket] = [self._socket]
        self._placeholder_sockets: List[socket.socket] = []

        self.nodes: Dict[Tuple[str, int], Node] = {}
        self._node_counter = 0
        self._new_nodes: Deque[int] = deque()
        self._disconnected_nodes: Deque[int] = deque()

    def close(self):
        """
        Shutdown and close the underlying UDP socket.

        NOTE: On Linux, `shutdown` throws an apparently avoidable error:
        'OSError [Errno 107] Transport endpoint is not connected'
        """

        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        self._socket.close()

    def sendall(self, data: bytes, address: Tuple[str, int]):
        """
        Ensure that the entirety of the data is sent
        (in chunks of limited size).

        NOTE: Joint packets could otherwise result in size that is
        unpredictably larger than the receiving buffer, but the latter
        must be large enough to be able to read full datagrams.
        """

        if len(data) > self._send_chunk_size:
            chunks = (data[i:i+self._send_chunk_size] for i in range(0, len(data), self._send_chunk_size))

        else:
            chunks = (data,)

        for chunk in chunks:
            len_sent = 0

            while len_sent != len(chunk):
                len_sent += self._socket.sendto(chunk[len_sent:] if len_sent != 0 else chunk, address)

    def send(self, timestamp: float = 0.) -> List[int]:
        """
        Send and clear buffered data for each client node after confirming its
        connection, returning the list of disconnected nodes.
        """

        self._disconnected_nodes.clear()

        for address, node in list(self.nodes.items()):
            try:
                node.verify_connection(timestamp)

            except ConnectionError:
                self._disconnected_nodes.append(node.id)
                del self.nodes[address]
                continue

            if node.outgoing_buffer:
                self.sendall(b''.join(node.outgoing_buffer), address)
                node.outgoing_buffer.clear()

        return self._disconnected_nodes

    def recv(self, timestamp: float = 0., blocklist: list[str] = None) -> List[Node]:
        """
        Check if data is available for reading, determine its source node,
        and receive it into its buffer. A successful receive also updates
        its connection status.

        NOTE: Apparently, an UDP packet not reaching its destination can cause an error (on Windows, at least). See:
        https://bobobobo.wordpress.com/2009/05/17/udp-an-existing-connection-was-forcibly-closed-by-the-remote-host/
        """

        self._new_nodes.clear()

        while True:
            readable, _, _ = select(self._readable_sockets, self._placeholder_sockets, self._placeholder_sockets, 0.)

            if readable:
                try:
                    data, address = self._socket.recvfrom(self._recv_chunk_size)

                # See note in docstring
                except ConnectionResetError:
                    continue

                if blocklist is not None and address[0] in blocklist:
                    continue

                # Drop incomplete data packets
                if len(data) % self._client_message_size:
                    if len(data) < self._client_message_size:
                        continue

                    data = data[:len(data) - len(data) % self._client_message_size]

                # Add new client node
                if address not in self.nodes:
                    if len(self.nodes) < self._MAX_NUM_CONCURRENT_CONNECTIONS:
                        node_id = self._node_counter
                        self._node_counter += 1

                        node = Node(
                            id_=node_id, address=address, tracker=MovingAverageTracker(self._latency_window_length))

                        self.nodes[address] = node
                        self._new_nodes.append(node)

                    else:
                        continue

                else:
                    node = self.nodes[address]

                # Separate individual packets
                if len(data) > self._client_message_size:
                    node.incoming_buffer.extend(
                        data[i:i+self._client_message_size] for i in range(0, len(data), self._client_message_size))
                else:
                    node.incoming_buffer.append(data)

                node.update_connection_status(timestamp)

            else:
                break

        return self._new_nodes
