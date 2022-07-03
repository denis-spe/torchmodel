# import library 
from dataclasses import dataclass

class Hyperparameter(object):
    """
    The class contains methods for setting
    the parameter of the model
    """
    def __init__(self):
        """
        Class Constructor for initializing
        class parameter
        Return: 
            NoneType object
        """
        self.param = dict()
        self.select = {}

    def Choice(self, name, options) -> any:
        """
        The method is used to set name of    
        parameter with its options pack the value
        from
        Args:
            name(str): Name of the parameter
            options(list): List of options
        Return: Any value from options
        """
        # import hidden torch libraries
        import torch
        
        # Set the name as key and options as values to param parameter of the class
        self.param[name] = torch.tensor(options)
        
        # Check if the name is not in select dictonary then set the name as key and give it 
        # a default value of 0
        if name not in self.select:
            self.select[name] = 0
        else:
            pass
        
        try:
            return options[self.select[name]]
        except IndexError:
            # if options receives a select value which is out range then assign it to 0
            self.select[name] = 0
            return options[self.select[name]]
            
    def Int(self, name: str, min_value: int, max_value: int, step: int = None) -> int:
        """
        This method sets the name of the 
        parameter with  values from min_value to 
        max_value and jump each step if the step
        parameter is not none
        Args:
            name (str): Name of the parameter
            min_value(int): The value to start from
            max_value(int):  Value to end with
            step (int): Range of value to jump

        Return: Integer
        """
        # import hidden torch libraries
        import torch
        
        # Construct options
        options = torch.tensor(range(min_value, max_value, step))
        
        # Set the name as key and options as values to param parameter of the class
        self.param[name] = options
        
        # Check if the name is not in select dictonary then set the name as key and give it 
        # a default value of 0
        if name not in self.select:
            self.select[name] = 0
        else:
            pass
    
        try:
            return options[self.select[name]]
        except IndexError:
            # if options receives a select value which is out range then assign it to 0
            self.select[name] = 0
            return options[self.select[name]]
            
    def Float(self, name: str, min_value: any, max_value: any) -> int:
       """
        This method sets the name of the 
        parameter with  values from min_value to 
        max_value and jump each step if the step
        parameter is not none
        Args:
            name (str): Name of the parameter
            min_value(float): The value to start from
            max_value(float):  Value to end with

        Return: Float
        """
       # import hidden torch libraries
       import numpy as np
        
        # Construct options
       options = np.linspace(min_value, max_value)
        
        # Set the name as key and options as values to param parameter of the class
       self.param[name] = options
        
        # Check if the name is not in select dictonary then set the name as key and give it 
        # a default value of 0
       if name not in self.select:
           self.select[name] = 0
       else:
           pass
    
       try:
           return options[self.select[name]]
       except IndexError:
           # if options receives a select value which is out range then assign it to 0
           self.select[name] = 0
           return options[self.select[name]]
        
    def selector(self, name) -> None:
           """
           The method adds 1 on the selected key value
            Args:
                name (str): The name used as key in select dictionary to add 1 on it's value
             Return: NoneType
           """
           if self.select[name] < len(self.param[name]):
               self.select[name] += 1
           
 
@dataclass
class Tuner:
           """
           The class contains methods for running 
           model while tuning it's parameters
           """
           # Initialize the parameters in a short way
            # without the use of __init__ function
           model: any
           hyperparameter: Hyperparameter
           max_epoch: int
           
           
           def study(self):
               for _ in range(self.max_epoch):
                   print(self.model())
                   print(self.model().optim)
                   for param_name in self.hyperparameter.param.keys():
                       self.hyperparameter.selector(param_name)
                   
           
           




    
    
    
    


