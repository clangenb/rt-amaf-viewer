/*F***************************************************************************
 * 
 * openSMILE - the Munich open source Multimedia Interpretation by 
 * Large-scale Extraction toolkit
 * 
 * This file is part of openSMILE.
 * 
 * openSMILE is copyright (c) by audEERING GmbH. All rights reserved.
 * 
 * See file "COPYING" for details on usage rights and licensing terms.
 * By using, copying, editing, compiling, modifying, reading, etc. this
 * file, you agree to the licensing terms in the file COPYING.
 * If you do not agree to the licensing terms,
 * you must immediately destroy all copies of this file.
 * 
 * THIS SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS,
 * IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST
 * INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE
 * OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL
 * ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS
 * DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
 * NEITHER TUM NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY
 * DAMAGES RELATED TO THE SOFTWARE OR THIS LICENSE AGREEMENT, INCLUDING
 * DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE
 * MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON.
 * ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE
 * THE SOFTWARE OR DERIVATIVE WORKS.
 * 
 * Main authors: Florian Eyben, Felix Weninger, 
 * 	      Martin Woellmer, Bjoern Schuller
 * 
 * Copyright (c) 2008-2013, 
 *   Institute for Human-Machine Communication,
 *   Technische Universitaet Muenchen, Germany
 * 
 * Copyright (c) 2013-2015, 
 *   audEERING UG (haftungsbeschraenkt),
 *   Gilching, Germany
 * 
 * Copyright (c) 2016,	 
 *   audEERING GmbH,
 *   Gilching Germany
 ***************************************************************************E*/


/*  openSMILE component:

example dataSink:
reads data from data memory and outputs it to console/logfile (via smileLogger)
this component is also useful for debugging

*/


#include <iocore/funcConsoleSink.hpp>
#include <iostream>

#define MODULE "cFuncConsoleSink"


SMILECOMPONENT_STATICS(cFuncConsoleSink)

SMILECOMPONENT_REGCOMP(cFuncConsoleSink)
{
  SMILECOMPONENT_REGCOMP_INIT

  scname = COMPONENT_NAME_CFUNCCONSOLESINK;
  sdescription = COMPONENT_DESCRIPTION_CFUNCCONSOLESINK;

  // we inherit cDataSink configType and extend it:
  SMILECOMPONENT_INHERIT_CONFIGTYPE("cDataSink")
  
  SMILECOMPONENT_IFNOTREGAGAIN(
    ct->setField("filename","The name of a text file to dump values to (this file will be overwritten, if it exists)",(const char *)NULL);
    ct->setField("lag","Output data <lag> frames behind",0,0,0);
    ct->setField("output_interval", "Outputs data every <output_interval> frames",0,0,0);
  )

  SMILECOMPONENT_MAKEINFO(cFuncConsoleSink);
}

SMILECOMPONENT_CREATE(cFuncConsoleSink)

//-----

cFuncConsoleSink::cFuncConsoleSink(const char *_name) :
  cDataSink(_name),
  fHandle(NULL),
  tick_no(0)
{
  // ...
}

void cFuncConsoleSink::fetchConfig()
{
  cDataSink::fetchConfig();
  
  filename = getStr("filename");
  SMILE_DBG(2,"filename = '%s'",filename);
  lag = getInt("lag");
  SMILE_DBG(2,"lag = %i",lag);
  output_interval = getInt("output_interval");
  SMILE_DBG(2,"output_interval = %i",output_interval);
}


/*
int cFuncConsoleSink::myConfigureInstance()
{
  return  cDataSink::myConfigureInstance();
  
}
*/


int cFuncConsoleSink::myFinaliseInstance()
{
  // FIRST call cDataSink myFinaliseInstance, this will finalise our dataWriter...
  int ret = cDataSink::myFinaliseInstance();

  /* if that was SUCCESSFUL (i.e. ret == 1), then do some stuff like
     loading models or opening output files: */

  if ((ret)&&(filename != NULL)) {
    fHandle = fopen(filename,"w");
    if (fHandle == NULL) {
      SMILE_IERR(1,"failed to open file '%s' for writing!",filename);
      COMP_ERR("aborting");
	    //return 0;
    }
  }

  // write header
  long _N = reader_->getLevelN();
  long i;
  for(i=0; i<_N-1; i++) {
    char *tmp = reader_->getElementName(i);
    printf("%s ", tmp);
    free(tmp);
  }
  char *tmp = reader_->getElementName(i);
  printf("%s\n", tmp);
  free(tmp);

  return ret;
}


int cFuncConsoleSink::myTick(long long t)
{
  SMILE_DBG(4,"tick # %i, reading value vector:",t);
  cVector *vec= reader_->getFrameRel(lag);  //new cVector(nValues+1);
  if (vec == NULL) return 0;
  //else reader->nextFrame();

  long vi = vec->tmeta->vIdx;
  double tm = vec->tmeta->time;

  // now print the vector:
  int i;

  if (tick_no % output_interval == 0) {
	  for (i=0; i<vec->N-1; i++) {
	      printf("%f ", vec->dataF[i]);
	  }
	  printf("%f\n", vec->dataF[i]);
  }

  if (tick_no == 100) {tick_no = 0;}
  tick_no++;

  if (fHandle != NULL) {
    for (i=0; i<vec->N; i++) {
      fprintf(fHandle, "%s = %f\n",vec->name(i),vec->dataF[i]);
    }
  }

  // tick success
  return 1;
}


cFuncConsoleSink::~cFuncConsoleSink()
{
  if (fHandle != NULL) {
    fclose(fHandle);
  }
}

