from pathlib import Path

import pdfkit

__name__ = 'generator'


class Generator:
    def __init__(self):
        pass

    def generate_pdf(self, html=str(Path(__file__).parent / 'data/test.html'), location='results.pdf', debug=False):
        '''Convert HTML file to PDF file
        Convert HTML to PDF, prints 'PDF generated' to the terminal when finished if debug mode is enabled.
        Uses pdfkit library.


        Parameters
        ----------
            html : str
                file path to the HTML file to be converted
            location : str
                file path to the location where the PDF should be saved
            debug : bool, optional
                use debug mode or not (default is False)
        '''
        path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        pdfkit.from_file(html, location, configuration=config)
        if debug:
            print('PDF generated')
