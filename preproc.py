from yolo_label import yolo_form

with open("labels_def1.txt","r") as file:
    counter=1
    while counter<=150:
        a=file.readline()
        a=a.split(" ")
        b=[]
        b_int=[]
        for i in a:
            if(i == ""):
                pass
            else:
                b.append(i)
            
        major_axis=float(b[1])
        minor_axis=float(b[2])
        ang=float(b[3])
        x_pos=float(b[4])
        y_pos=float(b[5])
        
        x,y,w,h=yolo_form(major_axis,minor_axis,ang,x_pos,y_pos)
        
        with open("raw/Class1_def/{}.txt".format(counter), "w") as f:
            f.write("0 {} {} {} {}".format(x,y,w,h))
    
        counter+=1
           

    