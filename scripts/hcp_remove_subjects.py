'''
hcp_remove_subjects:
Script for manipulating the HCP subjects, specifically for removing certain subjects from the list.
In this case it used to remove the subjects which are already used for training, validation or testing 
in the first case.

Also can be used randomly sample a certain number of subjects - to produce other training, validation or test sets. 
'''

import random

def remove_subs(all_subs, rm_subs):
    new_subs = list(set(all_subs) - set(rm_subs))
    return new_subs

path = 'D:\Diffusion MRI data\hcp_subject_list.txt'

with open(path, 'r') as f:
    subjects = f.read()

subject_list = '  '.join(subjects.split('\n')).split('  ')

train_subjects = ['100206',
'100307',
'100408',
'100610',
'101006',
'101107',
'101309',
'101410',
'101915',
'102311',
'102513',
'102614',
'102715',
'102816',
'103010',
'103111',
'103212',
'103414',
'103515',
'103818']

validation_subjects = ['104012',  # Subjects to be used for validation
'104416',
'104820']

test_subjects = ['145127',
'147737',
'174437',
'178849',
'318637',
'581450',
'130821']

if __name__ == '__main__':
    new_subs = remove_subs(subject_list, train_subjects)
    new_subs = remove_subs(new_subs, validation_subjects)
    new_subs = remove_subs(new_subs, train_subjects)

    new_training_set = random.sample(new_subs, k=2)
    print(new_training_set)