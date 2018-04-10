import csv

class Logger():
    def __init__(self, file_name="data.txt"):
        self.file_name = file_name
        
    def open(self, labels):
        self.csvfile = open(self.file_name, 'w')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(labels)
        
    def close(self):
        self.csvfile.close()
        
    def log(self, data):
        self.writer.writerow(data)