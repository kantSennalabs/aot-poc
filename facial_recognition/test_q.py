import requests

def test_q(file_name):
    print(file_name)
    f = open(f"{file_name}.txt", "a")
    f.write("Now the file has more content!")
    f.close()
    return "finished job"