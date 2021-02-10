# acLogging.py
#
# Copyright 2021 Ottawa Hospital Research Institute
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Revision history:
#   Original Python version by Andre Carrington, 2021

def findNextFileNumber(fnprefix,fnsuffix):
    from os.path import isfile

    # find output file 100's digit...
    for c in range(1, 10): # 1 to 9
        fnum = c*100
        fn   = f'{fnprefix}{fnum:03d}{fnsuffix}'
        if isfile(fn):
          continue
        else:
          c = c - 1
          break
        #endif
    #endif
    if c < 0:
        c = 0
    #endif

    # find output file 10's digit...
    for a in range(1, 10): # 1 to 9
        fnum = (c*100)+(a*10)
        fn   = f'{fnprefix}{fnum:03d}{fnsuffix}'
        if isfile(fn):
          continue
        else:
          a = a - 1
          break
        #endif
    #endif
    if a < 0:
        a = 0
    #endif

    # find output file 1's digit...
    for b in range(1, 10): # 1 to 9
        fnum = (c*100)+(a*10)+b
        fn   = f'{fnprefix}{fnum:03d}{fnsuffix}'
        if isfile(fn):
            if b==9 and a==9: # if 99 or 299
                c = c + 1     # then next is 100 or 300 
                b = 0
                a = 0
                break
            #endif
            if b==9:          # if 39 or 259
                a = a + 1     # then next is 40 or 260
                b = 0
                break
            #endif
            continue
        else:
            break
        #endif
    #endfor
    fnum = (c*100)+(a*10)+b
    fn = f'{fnprefix}{fnum:03d}{fnsuffix}'
    return fn, fnum
#enddef
