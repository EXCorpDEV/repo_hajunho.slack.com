//
//  jhGuideLine.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

class jhGuideLine : CALayer {
    
    var layer_size : CGSize = CGSize.init(width: 0, height: 0)
    var lineX : CGFloat
    var lineY : CGFloat
    var lineWidth : CGFloat
    
    init?(x: CGFloat, y: CGFloat, lineWidth: CGFloat, layer:Any) {
        if jhGS.s.logLevel.contains(.graph2) { print("jhGuideLine_init") }
        self.lineX = x
        self.lineY = y
        self.lineWidth = lineWidth
        super.init(layer: layer)
//        super.isGeometryFlipped = true
        super.backgroundColor = UIColor(red: 255, green: 255, blue: 255, alpha: 0).cgColor
        super.isHidden = false
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(in ctx: CGContext) {
        
        if jhGS.s.logLevel.contains(.graph2) { print("jhGuideLine_draw, layer_size.width = \(layer_size.width), layer_size.height = \(layer_size.height), lineX = \(lineX), lineY = \(lineY)") }
        
        layer_size = self.bounds.size
        ctx.move(to : CGPoint(x : lineX, y : lineY))
        ctx.addLine(to: CGPoint(x: lineX, y: UIScreen.main.bounds.height)) //TODO: will be changed to current Scene Size
        ctx.setLineWidth(lineWidth)
        ctx.setStrokeColor(UIColor(red: 0, green: 0, blue: 0, alpha: 1.0).cgColor)
//        ctx.setStrokeColor(UIColor(red: 0, green: 185, blue: 255, alpha: 1.0).cgColor)
        ctx.strokePath()
        
    }
    
}
