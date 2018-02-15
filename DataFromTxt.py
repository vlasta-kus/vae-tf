import sys
import pandas as pd
import numpy as np

class DataFromTxt:
    """ Read train/test data from .txt file """

    def __init__(self, path):
        self.path = path
        self.inputDf = pd.DataFrame({'lineIdx': [], 'vector': []})
        self.dimension = 0
        self.current_position = 0
        self.epochs_completed = 0

        with open(path, 'r') as f:
            idx = -1
            for line in f:
                idx += 1
                # ignore head (column names)
                if idx == 0:
                    continue
                vec = self.processLine(line, idx)
                if vec:
                    self.inputDf = self.inputDf.append({'lineIdx': idx, 'vector': np.array(vec)}, ignore_index=True)

        print(" Input DataFrame size is %d." % self.inputDf.lineIdx.size)

        # get prevailing vector dimension
        self.inputDf['dim'] = self.inputDf['vector'].apply(lambda vec: len(vec))
        self.dimension = np.argmax(self.inputDf.groupby('dim').lineIdx.count())
        print(" Prevailing vector dimension: %d" % self.dimension)
        
        self.finalInput = self.inputDf[self.inputDf.dim == self.dimension]
        self.n_documents = self.finalInput.lineIdx.count()
        print(" n_documents: %d" % self.n_documents)

        #npArray = np.vstack(self.inputDf[self.inputDf.dim == self.dimension].vector.values)

    def next_batch(self, batch_size):
        if not self.has_next():
            self.reset()
        end = self.current_position + batch_size
        if end > self.n_documents:
            end = self.n_documents
            self.epochs_completed += 1
        df = self.finalInput[self.current_position : end]
        self.current_position = end
        return self.dfToNumpy(df), None

    def has_next(self):
        return self.current_position < self.n_documents

    def reset(self):
        self.current_position = 0

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def getTrainData(self):
        return self.dfToNumpy(self.finalInput)

    def getTestData(self):
        print(" ERROR: getTestData() not implemented yet!")

    def dfToNumpy(self, df):
        return np.vstack(df.vector.values)

    def processLine(self, line, idx):
        if "[" not in line:
            print(" WARN: skipping line %d because it does not contain required delimiter")
            return None
        strVec = line.rstrip().split("[")[1][:-2]
        try:
            vec = [float(i) for i in strVec.split(",")]
        except Exception as e:
            print(" ERROR transforming input at line %d to a vector: cannot convert to float" % idx)
            return None
        return vec


if __name__ == "__main__":
    assert len(sys.argv) > 1, " ERROR: missing command-line argument (path to a data file)"
    data = DataFromTxt(sys.argv[1])
