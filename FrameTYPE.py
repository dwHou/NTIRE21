import os
import sys

file_lst = os.listdir("./fixedrate_mkv")


for file in file_lst:
    vid = os.path.join("./fixedrate_mkv", file)
    os.system(f"ffprobe -i {vid} -show_frames | grep 'pict_type' > {file}_type.txt")
    file_num = file[0:3]
    png_lst = os.listdir(f"./fixedrate_png/{file_num}")
    png_lst.sort()
    i = 0
    with open(f'{file}_type.txt') as vidfile:
        # type_lst = []
        for l in vidfile.readlines():
            idx = l.find('=')
            if idx == -1:
                continue

            t = l[idx+1:idx+2]
            if t == 'I':
                os.system(f"mv ./fixedrate_png/{file_num}/{png_lst[i]} ./fixedrate_png/{file_num}/{png_lst[i][:-4]}_{1.1}.png")
            elif t == 'P':
                os.system(f"mv ./fixedrate_png/{file_num}/{png_lst[i]} ./fixedrate_png/{file_num}/{png_lst[i][:-4]}_{1.0}.png")
            elif t == 'B':
                os.system(f"mv ./fixedrate_png/{file_num}/{png_lst[i]} ./fixedrate_png/{file_num}/{png_lst[i][:-4]}_{0.9}.png")
            else:
                pass
            i += 1


            # type_lst.append(l[idx+1:idx+2])
    # line = file + ' ' + ''.join(type_lst) + str(len(type_lst)) + '\n'
    # print(line)
    # print(len(line))
    # f.write(line)
    os.system(f"rm {file}_type.txt")
