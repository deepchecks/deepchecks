/*
 * Deepchecks
 * Copyright (C) 2021 Deepchecks
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Deepchecks',
  tagline: '',
  url: 'https://docs.deepchecks.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  trailingSlash: true,
  favicon: 'img/favicon.ico',
  organizationName: 'deepchecks', // Usually your GitHub org/user name.
  projectName: 'deepchecks', // Usually your repo name.
  stylesheets: [
    'https://fonts.googleapis.com/css2?family=Roboto:wght@500&display=swap',
  ],
  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: '',
        logo: {
          alt: 'Package Logo',
          src: 'img/logo-light.svg',
          srcDark: 'img/logo-dark.svg'
        },
        items: [
          {
            type: 'dropdown',
            label: 'Community',
            position: 'right',
            items: [
              {
                label: 'Slack',
                href: 'https://deepcheckscommunity.slack.com/',
              },
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/deepchecks',
              },
              {
                label: 'Github Issues',
                href: 'https://github.com/deepchecks/deepchecks/issues',
              },

            ]
          },
          // {
          //   type: 'docsVersionDropdown',
          //   position: 'left',
          // },
          {to: 'https://deepchecks.com/blog/', label: 'Blog', position: 'right'},
          {
            href: 'https://github.com/deepchecks/deepchecks',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'light',
        links: [
          // {
          //   title: 'Docs',
          //   items: [
          //     {
          //       label: 'Tutorial',
          //       to: '/docs/intro',
          //     },
          //   ],
          // },
          // {
          //   title: 'Community',
          //   items: [
          //     {
          //       label: 'Stack Overflow',
          //       href: 'https://stackoverflow.com/questions/tagged/deepchecks',
          //     },
          //     {
          //       label: 'Slack',
          //       href: 'https://discordapp.com/invite/docusaurus',
          //     },

              
          //   ],
          // },
          // {
          //   title: 'More',
          //   items: [
          //     {
          //       label: 'Blog',
          //       to: '/blog',
          //     },
          //     {
          //       label: 'GitHub',
          //       href: 'https://github.com/facebook/docusaurus',
          //     },
          //   ],
          // },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Deepchecks, Inc.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
