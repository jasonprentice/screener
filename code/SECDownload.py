from retrying import retry

@retry
def SECDownload(year, month, formType):
    # Download all XBRL reports of formType (10-Q or 10-K), filed in specified month and year.
    # Will populate folder ../data/edgar/

    from urllib2 import urlopen, HTTPError
    import xml.etree.ElementTree as ET
    import os
    import gzip
    
    urlString = 'http://www.sec.gov/Archives/edgar/monthly/xbrlrss-' + str(year) + '-' + str(month).zfill(2) + '.xml'
    
    #try:
    feedFile = urlopen( urlString )
    #        goodRead = False
    #        try:
    #            feedData = feedFile.read()
    #            goodRead = True
    #            print("Read successfully.")
    #        finally:

    tree = ET.parse(feedFile)
    feedFile.close()
    root = tree.getroot()
    namespaces = {'edgar': "http://www.sec.gov/Archives/edgar"}
    for item in root[0].findall('item'):
        if item.find('description').text == formType:
            title = item.find('title').text
            filing = item.find('edgar:xbrlFiling', namespaces=namespaces)
            cikNumber = filing.find('edgar:cikNumber', namespaces=namespaces).text
            name = filing.find('edgar:companyName', namespaces=namespaces).text
            name = name.upper()
            print(name + " " + str(year) + " " + str(month).zfill(2))

            base_dir = "../data/edgar/" + str(cikNumber)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            if not os.path.isfile(base_dir + "/company_name.txt"):
                file = open(base_dir + "/company_name.txt", 'wb')
                file.write(name)
                file.close()

            sub_dir = base_dir + "/" + str(year) + "/" + str(month).zfill(2)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            
    #                if not os.path.exists(
    #                if not os.path.exists("edgar/" + str(year)):
    #                    os.makedirs("edgar/" + str(year))
    #                if not os.path.exists("edgar/" + str(year) + "/" + str(month).zfill(2)):
    #                    os.makedirs("edgar/" + str(year) + "/" + str(month).zfill(2))
    #                
    #                base_dir = "edgar/" + str(year) + "/" + str(month).zfill(2) + "/"
    #
    #                sub_dir = base_dir + str(cikNumber)

            xbrlFiles = filing.find('edgar:xbrlFiles', namespaces=namespaces)
            for xbrlFile in xbrlFiles.findall('edgar:xbrlFile', namespaces=namespaces):
    #                    print xbrlFile.attrib
                downloadThis = False
                if xbrlFile.attrib['{'+namespaces['edgar']+'}'+'type'] == 'EX-101.INS':
                    instanceURL = xbrlFile.attrib['{'+namespaces['edgar']+'}'+'url']
                    fileName = formType
                    if not os.path.isfile(sub_dir + "/" + fileName + ".xml.gz"):
                        full_FileName = sub_dir + "/" + fileName + ".xml.gz"
                        downloadThis = True
                elif xbrlFile.attrib['{'+namespaces['edgar']+'}'+'type'] == 'EX-101.CAL':
                    instanceURL = xbrlFile.attrib['{'+namespaces['edgar']+'}'+'url']
                    fileName = formType + "-cal"
                    if not os.path.isfile(sub_dir + "/" + fileName + ".xml.gz"):
                        full_FileName = sub_dir + "/" + fileName + ".xml.gz"
    	             	downloadThis = True
                elif xbrlFile.attrib['{'+namespaces['edgar']+'}'+'type'] == formType:
                    instanceURL = xbrlFile.attrib['{'+namespaces['edgar']+'}'+'url']
                    fileName = formType + "-html"
                    if not os.path.isfile(sub_dir + "/" + fileName + ".html.gz"):
                        full_FileName = sub_dir + "/" + fileName + ".html.gz"
                        downloadThis = True
                if downloadThis:
                    instanceFile = urlopen(instanceURL)
                    instanceData = instanceFile.read()
                    file = gzip.open(full_FileName,'wb')
                    file.write(instanceData)
                    file.close()
                    instanceFile.close()

   # except HTTPError as e:
   #     print("HTTP Error: ", e.code)

#    if goodRead == True:


if __name__ == '__main__':

    for year in range(2010,2015):
        for month in range(1,13):
            SECDownload(year, month,'10-Q')
