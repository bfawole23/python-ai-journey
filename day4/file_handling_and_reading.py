# w means that it create the file if it does not exist
with open("notes.txt", "w") as file:

# with open(...) as file: is the standard pattern.it automatically close the file 
# forgetting to close the file cause real bugs
    file.write("Patient amaka checked in today. \n")
    file.write("Blood pressure normal. \n")
# \n means new line
# r means read mode


with open("notes.txt", "a") as file:
     file.write("follow up on this.\n")
    


with open("notes.txt", "r") as file:
    content = file.read()
    print(content)






