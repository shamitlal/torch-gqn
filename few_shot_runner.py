import numpy as np 
import torch 
import ipdb 
st = ipdb.set_trace

class few_shot_runner():
    def __init__(self, name, num_shots, num_categories):
        self.num_shots = num_shots
        self.num_categories = num_categories
        self.max_evaluations = 200
        self.summ_writer = None
        self.name = name
        self.reset()

    def reset(self):
        self.dict = {}
        self.num_evaluations = 0
        self.total = 0
        self.no_change_streak = 0
        self.correct = 0
        
    
    def is_full(self):
        if len(self.dict.keys()) < self.num_categories:
            print(f"Got {len(self.dict.keys())}/{self.num_categories} keys in {self.name} runner")
            return False

        for key in self.dict.keys():
            if len(self.dict[key]) < self.num_shots:
                return False 
        
        return True 

    def step(self, key, val, summ_writer, stepnum):
        self.summ_writer = summ_writer
        if self.is_full() or self.no_change_streak>100:
            print("Evaluating munit few shot")
            self.evaluate(key, val, stepnum)
        else:
            print("Filling munit few shot")
            self.store(key, val)

    def evaluate(self, key, val, stepnum):
        # st()
        if self.num_evaluations == 0:
            self.compile_info()
        
        if key not in self.dict.keys():
            return 

        best_cossim = -10
        best_key = None
        for storedkey in self.dict.keys():
            storedval = self.dict[storedkey]
            cossim = self.get_cosine_sim(storedval, val)
            if cossim > best_cossim:
                best_cossim = cossim
                best_key = storedkey
        
        if key == best_key:
            self.correct += 1.0
        self.total += 1.0
        print(f"Accuracy for {self.name}: ", self.correct/self.total)
        self.summ_writer.add_scalar(f"{self.name}_fewshot", self.correct/self.total, stepnum)

        self.num_evaluations += 1
        if self.num_evaluations >= self.max_evaluations:
            self.reset()

         
    def get_cosine_sim(self, val1, val2):
        # val -> C
        cossim = torch.sum(val1*val2)
        cossim = cossim/(val1.norm() + 1e-5)
        cossim = cossim/(val2.norm() + 1e-5)
        return cossim

    def store(self, key, val):
        if key not in self.dict:
            self.dict[key] = []

        if len(self.dict[key]) < self.num_shots:
            self.dict[key].append(val)
            self.no_change_streak = 0
        self.no_change_streak += 1


    
    def compile_info(self):
        for key in self.dict.keys():
            val = self.dict[key]
            val = torch.mean(torch.stack(val, dim=0), dim=0)
            self.dict[key] = val 
