#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd


# In[11]:


weight1 = 1.0
weight2 = 1.0
bias = -0.501


# In[12]:


test_inputs = [(0,0) , (0,1) , (1,1)]
correct_outputs = [False , False , False , True]
outputs = []


# In[13]:


for test_input , correct_output in zip(test_inputs , correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination>=0)
    is_correct_string = 'yes' if output == correct_output else 'no'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])


# In[14]:


num_wrong = len([output[4] for output in outputs if output[4]=='no'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', 'Input2', 'Linear Combination' , 'Activation Output', 'Is Correct'])
if not num_wrong:
    print('noice.\n')
else:
    print('galat\n'.format(num_wrong))
print(output_frame.to_string(index=False))

