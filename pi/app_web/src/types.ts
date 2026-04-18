export type Detection = {
  id: string;
  timestamp: string;
  probability: number;
  datasetType: 'TON_IoT' | 'eMBB' | 'mMTC' | 'URLLC';
}