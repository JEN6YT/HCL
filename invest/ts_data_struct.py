from bidict import bidict

class BiHashList:
    """
    A python list, whose indices are accessible through a bidirectional hash table. Custom python data structure to store time series.
    It is useful for retrieving a time range of data as a list. It can bidirectionally map keys to indices and vice versa.
    """
    def __init__(self):
        self._list = []
        self._bD = bidict({})
    
    def __getitem__(self, key):
        return self._list[self._bD[key]]
    
    def get_item_by_index(self, index):
        return self._list[index]
    
    def __len__(self):
        return len(self._list)
    
    def __setitem__(self, key, value):
        if key in self._bD:
            self._list[self._bD[key]] = value
        else:
            self.append(key, value)
        
    def append(self, key, value):
        self._list.append(value)
        self._bD[key] = len(self._list) - 1
    
    def __delitem__(self, key):
        if key not in self._bD:
            raise KeyError(f"Key {key} not in BiHashList")
        index = self._bD[key]
        del self._list[index]
        del self._bD[key]
        for k, v in self._bD.items():
            if v > index:
                self._bD[k] -= 1

    def del_item_by_index(self, index):
        if index not in self._bD.inv:
            raise KeyError(f"Index {index} not in BiHashList")
        key = self._bD.inv[index]
        del self._list[index]
        del self._bD[key]
        for k, v in self._bD.items():
            if v > index:
                self._bD[k] -= 1

    def __contains__(self, key):
        return key in self._bD
        
    def return_ranged_value_list_from_indices(self, start_index, end_index):
        return self._list[start_index:end_index+1]

    def return_ranged_value_list_from_keys(self, start_key, end_key):
        start_index = self._bD[start_key]
        end_index = self._bD[end_key]
        return self._list[start_index:end_index+1] 
        # note the +1, the range end key should exist, and +1 makes it possible to return the last element 