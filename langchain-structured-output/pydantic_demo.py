# basic example of pydantic
# pydantic hum type restrictions ke liye use krte hai agr kis variable ka type string h to usme hum intiger ni dal skte,, ye wala kam TypeDict library bhi krti hai lekin wo restriction ni lgati sirf batati hai ki ye string hona chhaiye or ye number agr phir user na maane toh wo koi error ni deti lekin pydantic error de deti h 

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None #ye use hota h optional values ke liye ki agr koi value na bhi de rhe ho toh uski jgh null aa jaye ga 
    email: EmailStr #ye ek stricted function h ki email ka format shi hi hona chahiye
    cgpa: float = Field(gt=0, lt=10)  #gt = greater than lt=less than. it use to fix the range 
    
#new_student = {'name':'32'} #it  dont give error because name is of string type

#.............Multi parameters..................
#new_student = {'name':'32', 'age':23} #it give error because name is of string type

# ...............Coercing.........................
#new_student = {'age':'36'} #agr hum age ko string me bhi bhejhte h toh pydantic itna smart hota h ki wo smjh jata h ki ye ek numeric value hai or iska type bhi int define hai upr toh automatichaly behind the scene isko numeric me convert kr deta h or hume error ni milta iss concept ko coercing bolte h

# .....................EmailStr.........................
#new_student = {'email':'abc@gmail.com'} #it dont give the error
#new_student = {'email':'abc'} #hence email is not in the correct format so it give the error


# ......................Field.......................
# new_student  = {'cgpa':12} #Xwrong 
# new_student  = {'cgpa':9} right


student = Student(**new_student)

print(student)
