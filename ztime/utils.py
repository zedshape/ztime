import csv

class Database:
    def __init__(self, database):
        self.sequences = []
        self.initialSupport = {}
        self.frequentSecondElements = set()

        for id, eSeqList in enumerate(database):
            newSeq = EventSequence(id)
            eventList = newSeq.processAdding(eSeqList)
            self.sequences.append(newSeq)

            for label in eventList:
                if label not in self.initialSupport:
                    self.initialSupport[label] = 0
                self.initialSupport[label] += 1
        
        for eSeq in self.sequences:
            eSeq.sequences.sort()

    def remove(self):
        for idx, seq in enumerate(self.sequences):
            for iidx, evt in enumerate(seq.sequences):
                if evt.label not in self.initialSupport.keys():
                    del self.sequences[idx].sequences[iidx]

    def __str__(self):
        rst = []
        for i, eSeq in enumerate(self.sequences):
            rst.append(format("eSeq %d : %s" % (i, eSeq.__str__())))
        return "\n".join(rst)


class EventSequence:
    def __init__(self, id):
        self.id = id
        self.sequences = []  # order of event

    def processAdding(self, eSeqList):
        eventList = set()
        for event in eSeqList:
            newInterval = Interval(event[0], event[1], event[2])
            self.sequences.append(newInterval)
            eventList.add(newInterval.label)
        self.sequences = sorted(self.sequences)
        return eventList

    def __repr__(self):
        rst = []
        for event in self.sequences:
            rst.append(event.__str__())
        return "(" + ", ".join(rst) + ")"

class Interval:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    def getDuration(self):
        return self.end - self.start

    def __hash__(self):
        return hash((self.label, self.start, self.end))

    def __repr__(self):
        return format("(%s, %d, %d)" % (self.label, self.start, self.end))

    def __lt__(self, other):
        if self.start == other.start:
            if self.end == other.end:
                return self.label < other.label
            else:
                return self.end < other.end
        else:
            return self.start < other.start


def getRelation(A, B):
    relation = 2 

    if B[0] - A[0] == 0:
        if B[1] - A[1] == 0:
            relation = 1 #equal
        else:
            relation = 2 #overlap
    elif B[0] - A[1] > 0:
        relation = 3 #follow

    return relation

def preprocess(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        your_list = list(reader)

    distinct_events = set()
    new_list = []
    final_list = []
    timelength = {}
    max_index = 0
    for i in your_list:
        new_list.append(i[0].split(" "))

    for i in new_list:
        max_index = max(int(i[0]), max_index)

    for i in range(max_index + 1):
        final_list.append([])

    for i in new_list:
        final_list[int(i[0])].append((str(i[1]), int(i[2]), int(i[3])))
        distinct_events.add(str(i[1]))
        if int(i[0]) not in timelength:
            timelength[int(i[0])] = 0
        timelength[int(i[0])] = max(timelength[int(i[0])], int(i[3]))

    tseq = len(final_list)
    tdis = len(distinct_events)
    tintv = len(new_list)
    aintv = len(new_list) / len(final_list)
    avgtime = sum(timelength.values()) / len(timelength.keys())

    return tseq, tdis, tintv, aintv, avgtime, final_list

def getEventIntervalSequences(z):
    return z[0]