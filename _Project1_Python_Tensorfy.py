#!/usr/bin/env python
# coding: utf-8

# In[63]:


import fnmatch
import os
def  FileSearching(directory, searchvalue):
  """Searches for a file with the given name/ extension  in the specified directory.

  Args:
    directory: The directory to search in.
    searchvalue: The name/extension  of the file to search for.

  Returns:
    The full path of the file if found, otherwise   
 None.
  """

  results = []
  for root, _, files in os.walk(directory):
    for file in files:
      if fnmatch.fnmatchcase(file, f"*{searchvalue}*") or file.endswith(f".{searchvalue}"):
        results.append(os.path.join(root, file))
  return results


# In[69]:


import re

def rename_file(file_path, new_name_rule):
  """Renames a file based on a given rule.

  Args:
    file_path: The full path of the file to be renamed.
    new_name_rule: A string representing the new name rule or regular expression.

  Returns:
    True if the file was renamed successfully, False otherwise.
  """

  try:
    file_dir, file_name = os.path.split(file_path)
    new_name = new_name_rule.format(file_name=file_name)  # Basic formatting
    new_name = re.sub(r'\{.*\}', lambda m: m.group()[1:-1], new_name)  # Regular expression substitution
    new_file_path = os.path.join(file_dir, new_name)
    os.rename(file_path, new_file_path)
    print('Done renaming')
  except Exception as e:
    print(f"Error renaming file: {e}")
    return False


# In[ ]:


file_path = r"C:\Users\hala mohamed\Desktop\2_test.tensorfy.txt"
new_name_rule = '{file_name}_renamed.tensorfy'  # Example new name rule

result = rename_file(file_path, new_name_rule)
print(f"Renaming result: {result}")


# In[ ]:


import tempfile #for creation of temp file
import shutil #high level operation on file

import tempfile
import shutil

def modify_file_content(file_path, find_text, replace_text):
    """
    Modifies the content of a file based on provided text to find and replace.

    Args:
        file_path: The full path of the file to be modified.
        find_text: The text to find in the file content.
        replace_text: The text to replace the found text with.

    Returns:
        A tuple (success, modified_content) where:
            - success (bool) is True if the file was modified successfully, False otherwise.
            - modified_content (str) is the content of the file after modification, or None if not successful.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Apply the modification
        modified_content = content.replace(find_text, replace_text)
        
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp()
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            temp_file.write(modified_content)

        # Replace the original file with the modified temporary file
        shutil.move(temp_path, file_path)

        print(f"File modified successfully: {file_path}")
        #return modified_content
    except Exception as e:
        print(f"Error modifying file: {e}")
        return False


# In[ ]:


#file_path=r"C:\Users\hala mohamed\Desktop\2_test.tensorfy_renamed.tensorfy"
#text1="hala"
#text2="student"
#print(modify_file_content(file_path,text1,text2))


# In[74]:


def get_user_input():
  sources = []
  while True:
    source = input("Enter source path (or 'q' to quit): ")
    if source == 'q':
      break
    sources.append(source)
  return sources


# In[76]:


from pathlib import Path

def file_operation(sources, dest, operation='copy'):
  """
  Performs file or directory operations (copy or move) between multiple sources and a destination.

  Args:
    sources: A list of source file or directory paths.
    dest: The destination directory path.
    operation: The operation to perform ('copy' or 'move'). Defaults to 'copy'.
  """

  for source in sources:
    source_path = Path(source)
    dest_path = Path(dest)

    if source_path.is_file():
        if dest_path.is_dir():
                dest_path = dest_path / source_path.name  # Create destination file path
        if operation == 'copy':
                shutil.copy(source, dest)
                print(f"Copied {source} to {dest}")
        elif operation == 'move':
                shutil.move(source, dest)
                print(f"Moved {source} to {dest}")
        else:
               raise ValueError("Invalid operation. Choose 'copy' or 'move'.")
    elif source_path.is_dir():
        if dest_path.is_file():
            raise ValueError("Destination must be a directory when copying or moving a directory.")
        if operation == 'copy':
            shutil.copytree(source, dest, dirs_exist_ok=True)
            print(f"Copied directory {source} to {dest}")
        elif operation == 'move':
            shutil.move(source, dest)
            print(f"Moved directory {source} to {dest}")
        else:
            raise ValueError("Invalid operation. Choose 'copy' or 'move'.")
    else:
      raise ValueError(f"Invalid source path: {source}")


# In[ ]:





# In[ ]:


source_folder = r"C:\Users\hala mohamed\Desktop\test3_tensorfy"
destination_folder = r"C:\Users\hala mohamed\Desktop\test1_tensorfy"
o='copy'

 

# Move a directory to another directory
file_operation(source_folder, destination_folder, o)


# In[80]:


while True:
    print('Welcome to the File Management Tool. Please select an option.')
    print('1-File Searching')
    print('2-Rename File')
    print('3-Modify File Content')
    print('4-Files Opration (copy/move)')
    print('5-exit')
    i=int(input())
    if(i==1):
        directory = input("Enter the directory to search: ")
        searchvalue = input("Enter the file name or extension to search for: ")
        results = FileSearching(directory, searchvalue)
        for i in results:
            print(i)
    elif(i==2):
        file=input("Enter your desired file path:")
        new=input("Enter your new name/rule: ")
        result = rename_file(file,new)
    elif (i==3):
        file=input("Enter your desired file path: ")
        t1=input("Enter the text you want to repalce it : ")
        t2=input("Enter The text to replace the found text with") 
        res=modify_file_content(file,t1,t2)
        
    elif(i==4):
         
        sources = get_user_input()
        dest = input("Enter destination path: ")
        operation = input("Enter operation (copy or move): ")
        file_operation(sources, dest, operation)
    else:
      print("Thank You for using our Tool!")
      break
        
    
    
        
        
    


# In[ ]:





# In[ ]:




