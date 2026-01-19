from __future__ import division, print_function
import numpy as np
import os
import time


class SampleDumpNPZ(object):
    def __init__(self, _dumpDir, _dumpLimit):
        if os.path.exists(_dumpDir):
            if not os.path.isdir(_dumpDir):
                raise IOError("_dumpDir '%s' is not a directory"%_dumpDir)
        else:
            os.makedirs(_dumpDir)
        self.filePath = os.path.join(_dumpDir, '{fileCount}.npz')
        self.fileCount = 0
        self.dumpLimit = _dumpLimit

    def __lshift__(self, samples):
        if self.dumpLimit <= 0: return
        ys, us = samples
        if len(ys) != len(us):
            raise RuntimeError("Number of Samples and number of Solutions differ: {} vs {}".format(len(ys), len(us)))
        fn = self.filePath.format(fileCount=self.fileCount)
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        print("{} Saving samples: '{}'".format(ts, fn))
        np.savez_compressed(fn, ys=ys, us=us)
        self.fileCount += 1
        self.dumpLimit -= len(ys)


class SampleDumpHDF5(object):
    def __init__(self, _dumpFile, _dumpLimit, _compression='lzf'):
        import h5py
        filePath = _dumpFile + ('' if _dumpFile.endswith('.h5') else '.h5')
        if os.path.exists(filePath):
            raise IOError("'%s' already exists"%filePath)

        self.dumped = 0
        self.__file = None
        self.__ys = None
        self.__us = None

        def null(self, samples): return

        def dump(self, samples):
            ys, us = samples
            if len(ys) != len(us):
                raise RuntimeError("Number of Samples and number of Solutions differ: {} vs {}".format(len(ys), len(us)))
            a = self.dumped
            d = min(len(ys), _dumpLimit-self.dumped)
            assert d > 0

            ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
            print("{} Saving samples {}-{}: '{}'".format(ts, a, a+d, filePath))
            self.__ys[a:a+d] = ys[:d]
            self.__us[a:a+d] = us[:d]
            self.dumped = a+d

            if self.dumped >= _dumpLimit:
                self.__file.close()
                self.__file = None
                self.__ys = None
                self.__us = None
                self.__dumpCmd = null

        def openFile(self, samples):
            self.__file = f = h5py.File(filePath)
            ys, us = samples
            self.__ys = f.create_dataset('ys', (_dumpLimit, ys.shape[1]), np.float64, chunks=ys.shape, compression=_compression)
            self.__us = f.create_dataset('us', (_dumpLimit, us.shape[1]), np.float64, chunks=us.shape, compression=_compression)
            self.__dumpCmd = dump

            self << samples

        self.__dumpCmd = openFile


    def __lshift__(self, samples):
        return self.__dumpCmd(self, samples)

    def __del__(self):
        if self.__file is not None:
            self.__ys.resize(self.dumped, axis=0)
            self.__us.resize(self.dumped, axis=0)
            self.__file.close()
