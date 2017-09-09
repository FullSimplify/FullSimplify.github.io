import falcon
import pandas as pd
import numpy as np


class welcomeClass(object):
    def on_get(self, req, resp):
        """ Welcome """
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')
        resp.set_header('content-type', 'text/html')

        doc = """
        <h1> Menu </h1>
        <h3><a href="/statistics/loads"> Loads Per Hour </a></h3>
        <h3><a href="/statistics/avg"> Average </a></h3>
        <h3><a href="/statistics/stddev"> Standard Deviation </a></h3>
        <h3><a href="/statistics/min"> Minimum </a></h3>
        <h3><a href="/statistics/max"> Maximum </a></h3>
        """
        resp.body = doc


class descriptionClass(object):
    """This class does parses the input file and prepares the data for the class resources
        One initial data loading allows to improve the execution time.
        """
    def __init__(self, data):
        # group by time
        self._timestamp = data.timestamp.groupby(pd.to_datetime(data.timestamp).dt.hour).count()
        self._hours = len(pd.unique(self._timestamp))
        self._hours_descr = self._timestamp.describe()

        # group by device
        self._devices = data.device_type.groupby(data.device_type).describe()
        self._devices_descr = data.device_type.groupby(data.device_type).count().describe()


chrono_button = """
<form method=get>
<input type='submit' value='anti-chronological'>
</form>
"""

anti_chrono_button = """
<form method=get>
<input type='submit' value='chronological'>
</form>
"""

class loadsClass(descriptionClass):
    def __init__(self, data):
        descriptionClass.__init__(self, data)
        self.loads_per_hour = None
        self._body = None
        self._button = chrono_button
        self._sorting = 'chrono'
        self.loads_per_hour = self._timestamp
        self._string_hours = None
        self._string_loads = None

    def loads(self):
        # prepare the Response Message
        # s = [str(i) + str(v) for i, v in zip(self.loads_per_hour.index, self.loads_per_hour.values)]
        # assert all(self.loads_per_hour)
        if self._sorting is 'chrono':
            self._string_hours = [str(i) for i in self.loads_per_hour.index]
            self._string_loads = [str(i) for i in self.loads_per_hour.values]
        elif self._sorting is 'anti_chrono':
            self._string_hours = [str(i) for i in self.loads_per_hour.index]
            self._string_loads = [str(i) for i in self.loads_per_hour.values]
            self._string_hours.reverse()
            self._string_loads.reverse()
        else:
            raise ValueError("Sorting not understood. sorting:", self._sorting)

        # prepare the HTML document
        doc = """<table style="width:30%;display:table-row-group;padding:5px"> 
        <tr> 
        <th>Hour&nbsp&nbsp&nbsp&nbsp</th> 
        <th>Loads per hour</th> 
        </tr> 
        <tr> 
        """
        for i in range(len(self.loads_per_hour.index)):
            doc = doc + "<td>" + self._string_hours[i] + "</td>" + "<td>" + self._string_loads[i] + "</td> </tr>"

        doc = doc + "change to " + self._button + "order"
        self._body = doc
        return self._body

    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')
        resp.set_header('content-type', 'text/html')

        # We avoid doing the calculations twice
        if self._body is None:
            try:
                # self._sorting = 'chrono'
                resp.body = self.loads()
            except ValueError:
                print("Error in self.loads(). Perhaps the input data has not been parsed successfully")
        elif self._sorting is 'chrono':
            self._sorting = 'anti_chrono'
            self._button = anti_chrono_button
            resp.body = self.loads()
        elif self._sorting is 'anti_chrono':
            self._sorting = 'chrono'
            self._button = chrono_button
            resp.body = self.loads()


class avgClass(descriptionClass):
    def __init__(self, data):
        descriptionClass.__init__(self, data)
        self.avg_loads_hour = None
        self.averages = None
        self._body = None

    def avg(self):
        # we want to calculate the average loads per hour
        try:
            self.avg_loads_hour = self._hours_descr['mean']
        except AttributeError:
            print("avgClass: something went wrong accessing _hours_descr from descriptionClass")

        # This Feature is Not Requested. We calculate also the averages per device type
        try:
            devices = np.append(self._devices.freq.index, 'total')
            avg_loads_device = self._devices.freq.values / self._hours
            # put all the averages together for print/display
            avg_values = np.append(avg_loads_device, self.avg_loads_hour)
            self.averages = np.vstack((devices, avg_values))
        except AttributeError:
            print("avgClass: something went wrong accessing _devices from descriptionClass")

        # prepare the Response Message
        s = [self.averages[0][i] + "\t\t\t" +
             str(int(np.around(self.averages[1][i]))) for i in range(len(self.averages) + 1)]
        delimiter = "\n"
        h = "Device type\tAverage loads per hour\n"
        line = str('-' * len(h) + "----" + "\n")
        self._body = (h + line + delimiter.join(s))
        return self.averages

    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

        # check if we already have the Response Message
        if self._body is None:
            # We avoid doing the calculations twice
            if self.averages is None:
                self.averages = self.avg()
                resp.body = self._body
            else:
                raise AttributeError("The average has not been calculated. "
                                     "Possibly the input data is in a non understood format.")
        else:
            resp.body = self._body


class stddevClass(descriptionClass):
    def __init__(self, data):
        descriptionClass.__init__(self, data)
        self._stddev_tot = None
        self._stddev_devices = None
        self.standard_deviation = None
        self._body = None

    def stddev(self):
        # check if we have read the file. If not we didn't make it to parse 'data' in the superclass
        try:
            self._stddev_tot = self._hours_descr['std']
            self._stddev_devices = self._devices_descr['std']

            self.standard_deviation = np.vstack((('time', 'devices'),
                              (np.around(self._stddev_tot, decimals=1),
                               np.around(self._stddev_devices, decimals=1)))).transpose()
        except AttributeError:
            print("stddevClass: something went wrong accessing the superclass' attributes")

        # prepare the Response Message
        s = [self.standard_deviation[i][0] + "\t\t\t" +
             self.standard_deviation[i][1] for i in range(len(self.standard_deviation))]
        delimiter = "\n"
        h = "factor \t\t Standard Deviation\n"
        line = str('-' * len(h) + "----" + "\n")
        self._body = (h + line + delimiter.join(s))
        return self.standard_deviation

    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

        # check if we already have the Response Message
        if self._body is None:
            # We avoid doing the calculations twice
            if self.standard_deviation is None:
                self.standard_deviation = self.stddev()
                resp.body = self._body
            if self.standard_deviation is None:
                raise AttributeError("The standard deviation has not been calculated. "
                                     "Possibly the input data is in a non understood format.")
        else:
            resp.body = self._body


class minClass(descriptionClass):
    def __init__(self, data):
        descriptionClass.__init__(self, data)
        self.minimum = None
        self._body = None

    def min(self):
        try:
            self.minimum = [self._timestamp.idxmin(), self._timestamp.min()]
        except AttributeError:
            print("minClass: something went wrong accessing the superclass' attributes")

        # prepare the Response Message
        s = str(self.minimum[0]) + "\t\t" + str(self.minimum[1])
        h = "Hour \t" + "      " + "Minimum \n"
        line = str('-' * len(h) + "\n")
        self._body = (h + line + str(s))
        return self.minimum

    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

        # check if we already have the Response Message
        if self._body is None:
            # We avoid doing the calculations twice
            if self.minimum is None:
                self.minimum = self.min()
                resp.body = self._body
            if self.minimum is None:
                raise AttributeError("The minimum has not been calculated. "
                                     "Possibly the input data is in a non understood format.")
        else:
            resp.body = self._body


class maxClass(descriptionClass):
    def __init__(self, data):
        descriptionClass.__init__(self, data)
        self.maximum = None
        self._body = None

    def max(self):
        try:
            self.maximum = [self._timestamp.idxmax(), self._timestamp.max()]
        except AttributeError:
            print("minClass: something went wrong accessing the superclass' attributes")

        # prepare the Response Message
        s = str(self.maximum[0]) + "\t\t" + str(self.maximum[1])
        h = "Hour \t" + "       " + "Maximum \n"
        line = str('-' * len(h) + "\n")
        self._body = (h + line + str(s))
        return self.maximum

    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.set_header('Powered-By', 'X-Service-Version: v0.1.0')

        # We avoid doing the calculations twice
        if self._body is None:
            if self.maximum is None:
                self.maximum = self.max()
                resp.body = self._body
            if self.maximum is None:
                raise AttributeError("The maximum has not been calculated. "
                                     "Possibly the input data is in a non understood format.")
        else:
            resp.body = self._body


# falcon WSGI app
app = falcon.API()

# Resources are represented by long-lived class instances
filePath = 'data.csv'
file = pd.read_csv(filePath, delimiter=',', nrows=1000000, engine='c')
mainResource = descriptionClass(file)
welcomeResource = welcomeClass()
loadsResource = loadsClass(file)
avgResource = avgClass(file)
stddevResource = stddevClass(file)
minResource = minClass(file)
maxResource = maxClass(file)

# handle all requests to the URL paths
app.add_route('/statistics', welcomeResource)
app.add_route('/statistics/loads', loadsResource)
app.add_route('/statistics/avg', avgResource)
app.add_route('/statistics/stddev', stddevResource)
app.add_route('/statistics/min', minResource)
app.add_route('/statistics/max', maxResource)
