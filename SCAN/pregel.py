import collections
import threading


class Vertex():
    def __init__(self, id, value, neighbors):
        # This is mostly self-explanatory, but has a few quirks:
        #
        # self.id unique ids for vertex
        # Each vertex stores the current superstep number in
        # self.superstep. It's arguably not wise to store many copies
        # of global state in instance variables, but Pregel's
        # synchronous nature lets us get away with it.
        self.id = id
        self.value = value
        self.neighbors = neighbors
        self.incoming_messages = []
        self.outgoing_messages = []
        self.active = True
        self.superstep = 0


class Pregel():

    def __init__(self, vertices, num_workers):
        self.vertices = vertices
        self.num_workers = num_workers

    def run(self):
        """Runs the Pregel instance."""
        self.partition = self.partition_vertices()
        while self.check_active():
            self.superstep()
            self.redistribute_messages()

    def partition_vertices(self):
        """Returns a dict with keys 0,....,self.num_workers-1
            representing the worker threads. The corresponding values are
            lists of vertices assigned to that worker.
        """
        partition = collections.defaultdict(list)
        for vertex in self.vertices:
            partition[self.worker(vertex)].append(vertex)
        return partition

    def worker(self, vertex):
        """Returns the id of the worker that vertex is assigned to."""
        return hash(vertex) % self.num_workers

    def superstep(self):
        workers = []
        for vertex_list in self.partition.values():
            worker = Worker(vertex_list)
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

    def redistribute_messages(self):
        """Updates the message lists for all vertices."""
        for vertex in self.vertices:
            vertex.superstep += 1
            vertex.incoming_messages = []
        for vertex in self.vertices:
            for(receiving_vertix, message) in vertex.outgoing_messages:
                receiving_vertix.incoming_messages.append((vertex, message))

    def check_active(self):
        """ Returns True if there are any active vertices, and False
             otherwise
        """
        return any([vertex.active for vertex in self.vertices])


class Worker(threading.Thread):

    def __init__(self, vertices):
        threading.Thread.__init__(self)
        self.vertices = vertices

    def run(self):
        self.superstep()

    def superstep(self):
        """Completes a single superstep for all the vertices in self."""
        for vertex in self.vertices:
            vertex.update()
