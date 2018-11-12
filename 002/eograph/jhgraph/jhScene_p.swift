//
//  graph_p.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

protocol jhScene_p {
    var jhGraphFrameWidth : UInt32 { get set }
    var jhGraphFrameHeight : UInt32 { get set }
    
    func drawBackboard()
    func drawBackboardLabels()
    func drawGraph()
}
