from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
        this function will return all the requirements
    '''
    requirements=[]
    with open(file_path)as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
     name='lung_cancer_prediction',
     version='0.0.1',
     author="Khushi",
     author_email="khushirajpurohit617@gmail.com",
     packages=find_packages(),
     install_requirements=get_requirements('requirements.txt')


)
