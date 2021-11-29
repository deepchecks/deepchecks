import React from 'react';
import  { Redirect } from 'react-router-dom';
import config from '../../docusaurus.config.js'

export default function Home() {
  return <Redirect to={config.baseUrl+"docs/intro"} />;
}