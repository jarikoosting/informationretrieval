def main():
    movies = open('movies.csv', 'r')
    outfile = open('moviesscore.csv', 'w')
    for line in movies:
        #print(line)
        linelist = line.rstrip().split(';')
        if linelist[2] == '':
            linelist[2] = 0
        elif int(linelist[2]) < 55:
            linelist[2] = 0
        else:
            linelist[2] = 1

        if int(linelist[0]) < 1000:
            outfile.write(linelist[0] + ';' + linelist[1] + ';' + str(linelist[2]) + '\n')
        else:
            outfile.write(linelist[0] + ';' + linelist[1] + ';' + str(linelist[2]))
    movies.close()
    outfile.close()


if __name__ == "__main__":
    main()
