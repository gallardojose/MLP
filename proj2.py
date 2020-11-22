import re

def read_file(filename):
    # feature_vecs = [ [id, vector, label], ... ]
    #example feature vecs = [ [2, [91,34,45,34], 5], [7, [45,33,78,12], 9], ... ]
    
    feature_vecs = []; 
    file = open(filename, "r")
    
    for line in file:
        vectors = []
        data = re.split(r'[()\s]\s*', line)
        while '' in data:
            data.remove('')
        
        for item in data[1:-1]:
            vectors.append(int(item))
        
        feature_vecs.append([int(data[0]), vectors , int(data[-1])])
        
    return feature_vecs



classified_set = read_file("ClassifiedSetData.txt")
