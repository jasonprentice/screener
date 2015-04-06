def parseLinkbase( linkbaseFiles ):
    # Calculation entries listed later in linkbaseFiles override those listed earlier
    
    from lxml import etree as ET
    import numpy as np
    
    allrows = {}
    tags = set()
    for linkbaseFile in linkbaseFiles:
        file_content = openFile(linkbaseFile)
        
        root = ET.fromstring(file_content)
        ns = root.nsmap

        if 'link' in ns:
            calcLinkTag = 'link:calculationLink'
            locTag = 'link:loc'
            arcTag = 'link:calculationArc'
        else:
            calcLinkTag = '{'+ns[None] +'}calculationLink'
            locTag = '{'+ns[None] +'}loc'
            arcTag = '{'+ns[None] +'}calculationArc'

        calculationLinks = root.findall(calcLinkTag, namespaces=ns)
        for calculationLink in calculationLinks:
            nodes = {}
            tag_index = {}
            n = 0
            for item in calculationLink.findall(locTag, namespaces=ns):
                tag = item.attrib['{'+ns['xlink']+'}href']
                tag = tag[tag.find('#')+1:]
                label = item.attrib['{'+ns['xlink']+'}label']
                nodes[label] = tag
                tag_index[tag] = n
                n = n+1

            rows = {}
            nrows = 0
            for item in calculationLink.findall(arcTag, namespaces=ns):
                weight = float(item.attrib['weight'])
                fromNode = item.attrib['{'+ns['xlink']+'}from'];
                toNode = item.attrib['{'+ns['xlink']+'}to'];
                key = nodes[fromNode]
                if key in rows:
                    rows[key].append( (nodes[toNode], weight) )
                else:
                    rows[key] = [(nodes[toNode], weight)]
                    nrows = nrows+1

            for key in rows:
                allrows[key] = rows[key]
            for key,val in nodes.iteritems():
                tags.add(val)

    return (allrows, tags)

def openFile(file):
    import gzip

    ftype = file[file.rfind('.')+1:]
    if ftype == 'gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'rb')
    file_content = f.read()
    f.close()
    return file_content

def parseStatement( statementFile, year ):
    #    import xml.etree.ElementTree as ET
    from lxml import etree as ET

    dir = statementFile[0:statementFile.find('10-K')]
    

    root = ET.fromstring(openFile(statementFile))
    ns = root.nsmap

    contextName = getInstantContext(root,year)
    print contextName

    bsTargets = ['Assets', 'AssetsCurrent', 'AssetsNoncurrent', 'CashCashEquivalentsAndShortTermInvestments', 'CashAndCashEquivalentsAtCarryingValue', 'ReceivablesNetCurrent', 'InventoryNet', 'InventoryNoncurrent', 'PropertyPlantAndEquipmentNet', 'PropertyPlantAndEquipmentGross', 'AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment', 'LongTermInvestmentsAndReceivablesNet', 'Goodwill', 'IntangibleAssetsNetExcludingGoodwill', 'LiabilitiesAndStockholdersEquity', 'Liabilities', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'LiabilitiesCurrent', 'LiabilitiesNoncurrent', 'AccountsPayableAndAccruedLiabilitiesCurrent', 'DebtCurrent', 'ShortTermBorrowings', 'LongTermDebtAndCapitalLeaseObligationsCurrent','LongTermDebtCurrent', 'CapitalLeaseObligationsCurrent', 'LongTermDebtAndCapitalLeaseObligations', 'LongTermDebtNoncurrent', 'CapitalLeaseObligationsNoncurrent', 'LiabilitiesOtherThanLongtermDebtNoncurrent', 'PreferredStockValue', 'CommonStockValue', 'AdditionalPaidInCapital', 'RetainedEarningsAccumulatedDeficit', 'MinorityInterest']


    linkbaseFiles = {'bs':['./us-gaap-2014-01-31/stm/us-gaap-stm-sfp-cls-cal-2014-01-31.xml', dir+'10-K-cal.xml.gz'], 'cf':'./us-gaap-2014-01-31/stm/us-gaap-stm-scf-indir-cal-2014-01-31.xml', 'is':'./us-gaap-2014-01-31/stm/us-gaap-stm-soi-cal-2014-01-31.xml'}
#    linkbaseFiles = {'bs':[dir+'10-K-cal.xml.gz'], 'cf':'./us-gaap-2014-01-31/stm/us-gaap-stm-scf-indir-cal-2014-01-31.xml', 'is':'./us-gaap-2014-01-31/stm/us-gaap-stm-soi-cal-2014-01-31.xml'}

    (rows,tags) = parseLinkbase(linkbaseFiles['bs'])

    nan = float('NaN')
    tagVals = {}
    for val in tags:
        tag = 'us-gaap:' + val[len('us-gaap_'):]
        #        tag = 'us-gaap:' + val[val.find('us-gaap_'):]
        tagVals[val] = nan
        node = root.find(".//" + tag + "[@contextRef='" + contextName + "']", namespaces=ns)
        if not node == None:
            tagVals[val] = node.text
        if tagVals[val] == None:
            tagVals[val] = 0

#        print tag + ": " + str(tagVals[val])
#        else:
#            print "No item " + tag

    tagVals = inferMissingValues(tagVals, rows)

    for target in bsTargets:
        tag = "us-gaap_" + target
        print target + ": " + str(tagVals[tag])
#        node = root.find(".//" + tag + "[@contextRef='" + contextName + "']", namespaces=ns)
#        if not node == None:
#            tagVals[tag] = node.text
#        else:
#            print "No item " + target


def inferMissingValues( tagVals, calcRows ):
    import math
    #    nan = float('NaN')
    for tag in tagVals:
        #        print tagVals[tag]
        if math.isnan(float(tagVals[tag])):# == 'nan':
            if tag == 'us-gaap:AccountsPayableAndAccruedLiabilitiesCurrent':
                tagVals = inferValueInner(tag, tagVals, calcRows,True)
            else:
                tagVals = inferValueInner(tag, tagVals, calcRows)
    return tagVals

def inferValueInner( tag, tagVals, calcRows, printFlag = False ):
    import math
    #    nan = float('NaN')
    if tag in calcRows:
        sum = 0;
        for child in calcRows[tag]:
            child_val = float(tagVals[child[0]])
            if math.isnan(child_val):# == nan:
                tagVals = inferValueInner(child[0],tagVals,calcRows,printFlag)
            
            child_val = float(tagVals[child[0]])
            if printFlag:
                print child[0] + ": " + str(child_val)

            sum = sum + child[1]*child_val
        tagVals[tag] = sum
    elif math.isnan(float(tagVals[tag])):# == nan:
        tagVals[tag] = 0

    return tagVals


def getInstantContext( root,year ):
    ns = root.nsmap
    for tag in root.findall('us-gaap:Assets', namespaces=ns):
        context = tag.attrib['contextRef']
        for ctxt in root.find("./xbrli:context[@id='"+context+"']", namespaces=ns):
            datenode = ctxt.find('.//xbrli:instant', namespaces=ns)
            if not datenode == None:
                datestr = datenode.text
                yr = datestr[0:datestr.find('-')]
                if yr == year and ctxt.find('./xbrli:entity/xbrli:segment/xbrldi:explcitMember', namespaces=ns) == None:
                    contextName = context
    
    return contextName



if __name__ == '__main__':
    
#    linkbaseFiles = {'bs':'./us-gaap-2014-01-31/stm/us-gaap-stm-sfp-cls-cal-2014-01-31.xml', 'cf':'./us-gaap-2014-01-31/stm/us-gaap-stm-scf-indir-cal-2014-01-31.xml', 'is':'./us-gaap-2014-01-31/stm/us-gaap-stm-soi-cal-2014-01-31.xml'}
#
#    parseLinkbase(linkbaseFiles['bs'])

    statementFile = './edgar/0000004515/2011/02/10-K.xml.gz'

    parseStatement( statementFile, '2010' )