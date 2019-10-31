import pandas as pd 


class svm():

    def getClasses(self, data):
        cl = list(set(data["class"]))
        result = dict()
        self.numClasses = len(cl)
        for i in range(self.numClasses):
            result[i] = cl[i]
        return result

    def processData(self, data, class_dictionary):
        for i in range(self.numClasses):
            name = "class_"+ str(i) +"_data"
            setattr(svm,name,data.loc[data["class"] == str(class_dictionary[i])].drop(["class"], axis=1))
            
   
    def output(self,class_dictionary):
        for i in range(self.numClasses):
            location = str('/home/luke/ml_ct4101/assignment2/data_split/' + class_dictionary[i] + '_data.csv')
            name = "class_"+ str(i) +"_data"
            getattr(self,name).to_csv(location, index=None)

if __name__ == '__main__':
    s = svm()
    data = pd.read_csv("hazelnuts.csv")
    class_dictionary = s.getClasses(data)
    s.processData(data,class_dictionary)
    s.output(class_dictionary)
   