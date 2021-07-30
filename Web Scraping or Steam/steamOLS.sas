proc import out = steam
	datafile="/home/u47321185/merged.csv" 
	dbms=csv replace;
	/*getnames=yes; datarow=2;*/
	
data steamTwo;
	set steam;
	
	if hasMult = "TRUE" then mult = 1;
	else mult = 0;
	
	if sameDevPub = "TRUE" then sameDP = 1;
	else sameDP = 0;
	
	if owners = "0 .. 20,000" then ownNum = 0;
	else if owners = "50,000 .. 100,000" then ownNum = 1;
	else if owners = "100,000 .. 200,000" then ownNum = 2;
	else if owners = "200,000 .. 500,000" then ownNum = 3;
	else if owners = "500,000 .. 1,000,000" then ownNum = 4;
	else if owners = "1,000,000 .. 2,000,000" then ownNum = 5;
	else if owners = "2,000,000 .. 5,000,000" then ownNum = 6;
	else if owners = "5,000,000 .. 10,000,000" then ownNum = 7;
	else if owners = "10,000,000 .. 20,000,000" then ownNum = 8;
	else ownNum = 9;
	
proc print;
	
proc reg data = steamTwo;
	model scores = ownNum initialprice discount percentPos mult numLang avMedDiff sameDP;
title1 'Score Correaltion';

run;
quit;
