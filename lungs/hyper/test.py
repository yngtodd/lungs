counter = 0

def myfun(iteration):
    print(f'Iteration {iteration}, counter: {counter}')
    counter = 0
    counter += 1


def main():
    for i in range(10):
        myfun(i)


if __name__=="__main__":
    main()
