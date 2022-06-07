import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt


def create_pdf(benign, poison, tp, fp, name):

    # Start the pdf for the report
    pdf = matplotlib.backends.backend_pdf.PdfPages(name)

    #create pdf
    plt.figure()
    plt.bar((10, 25, 33, 50, 75), benign[1], align='center', alpha=0.5, width=5)
    plt.ylabel('Accuracy')  
    plt.xlabel('Percentage of poisoned data')
    plt.title('Accuracy on benign data with pattern trigger')
    plt.ylim(90,100)
    plt.savefig(pdf, format='pdf')

    plt.figure()
    plt.bar((10, 25, 33, 50, 75), poison[1], align='center', alpha=0., width=5)
    plt.ylabel('Accuracy')  
    plt.xlabel('Percentage of poisoned data')
    plt.title('Accuracy on poisoned data with pattern trigger')
    plt.savefig(pdf, format='pdf')

    plt.figure()
    plt.bar((10, 25, 33, 50, 75), benign[1], align='center', alpha=0.5, width=5)
    plt.ylabel('Accuracy')  
    plt.xlabel('Percentage of poisoned data')
    plt.title('Accuracy on benign data with pixel trigger')
    plt.ylim(90,100)
    plt.savefig(pdf, format='pdf')

    plt.figure()
    plt.bar((10, 25, 33, 50, 75), poison[1], align='center', alpha=0.5, width=5)
    plt.ylabel('Accuracy')  
    plt.xlabel('Percentage of poisoned data')
    plt.title('Accuracy on poisoned data with pixel trigger')
    plt.savefig(pdf, format='pdf')

    i=0
    for x in (10, 25, 33, 50, 75):
        ++i
        plt.figure()
        plt.bar((10, 50, 100), fp[i], align='center', alpha=0.5, width=20)
        plt.ylabel('Clean images used in STRIP')  
        plt.xlabel('Percentage of false positives')
        plt.title('False poitives on ' + str(x) + '% poison')
        plt.savefig(pdf, format='pdf')

        plt.figure()
        plt.bar((10, 50, 100), tp[i], align='center', alpha=0.5, width=20)
        plt.ylabel('Clean images used in STRIP')  
        plt.xlabel('Percentage of true positives')
        plt.title('True poitives on ' + str(x) + '% poison')
        plt.savefig(pdf, format='pdf')


    pdf.close()
